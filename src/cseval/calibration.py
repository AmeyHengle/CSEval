from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .aspects import SPECS, Aspect
from .data import Item
from .evaluation import Level, correlate
from .llm import LLM, Request
from .parsing import parse_score
from .prompts import drafting_messages, refinement_messages, scoring_messages


@dataclass(frozen=True)
class CalibrationConfig:
    n_trials: int = 10
    shot_sizes: tuple[int, ...] = (3,)
    top_k: int = 3
    draft_temperature: tuple[float, float] = (0.0, 0.5)
    refine_temperature: float = 0.5
    scoring_temperature: float = 0.0
    misaligned_pool: int = 3
    misaligned_shots: int = 3
    eval_size: int | None = None
    selection_level: Level = "dataset"
    seed: int = 13
    draft_max_tokens: int = 1024
    score_max_tokens: int = 64

    def calls(self, n_eval: int) -> int:
        candidates = len(self.shot_sizes) * self.n_trials + self.top_k * self.n_trials
        return candidates + candidates * n_eval

    @classmethod
    def from_dict(cls, values: dict) -> CalibrationConfig:
        values = dict(values)
        for key in ("shot_sizes", "draft_temperature"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


@dataclass
class Candidate:
    id: str
    origin: str
    parent: str | None
    cot: str
    shots: list[str]
    temperature: float
    spearman: float
    kendall: float
    n_valid: int
    scores: dict[str, int | None] = field(repr=False)

    @property
    def fitness(self) -> float:
        return -math.inf if math.isnan(self.spearman) else self.spearman


@dataclass
class CalibrationResult:
    aspect: Aspect
    model: str
    config: CalibrationConfig
    data_fingerprint: str
    eval_uuids: list[str]
    candidates: list[Candidate]
    best_id: str

    @property
    def best(self) -> Candidate:
        return next(c for c in self.candidates if c.id == self.best_id)

    def save(self, path: Path | str) -> None:
        payload = asdict(self)
        payload["aspect"] = self.aspect.value
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> CalibrationResult:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["aspect"] = Aspect(payload["aspect"])
        payload["config"] = CalibrationConfig.from_dict(payload["config"])
        payload["candidates"] = [Candidate(**c) for c in payload["candidates"]]
        return cls(**payload)


def most_misaligned(candidate: Candidate, items: Sequence[Item], aspect: Aspect, size: int) -> list[tuple[Item, int]]:
    scored = [(item, candidate.scores[item.uuid]) for item in items if candidate.scores.get(item.uuid) is not None]
    scored.sort(key=lambda pair: abs(pair[1] - pair[0].gold[aspect]), reverse=True)
    return scored[:size]


def calibrate(
    llm: LLM,
    calibration_items: Sequence[Item],
    aspect: Aspect,
    config: CalibrationConfig,
    data_fingerprint: str = "",
    progress: Callable[[Candidate], None] | None = None,
) -> CalibrationResult:
    spec = SPECS[aspect]
    rng = random.Random(f"{config.seed}:{aspect.value}")
    usable = [item for item in calibration_items if item.gold[aspect] is not None]
    if config.eval_size is not None and config.eval_size < len(usable):
        eval_items = rng.sample(usable, config.eval_size)
    else:
        eval_items = list(usable)
    gold = [item.gold[aspect] for item in eval_items]
    groups = [item.hs_id for item in eval_items]
    candidates: list[Candidate] = []

    def evaluate(cot: str, origin: str, parent: str | None, shots: Sequence[Item], temperature: float) -> Candidate:
        scores: dict[str, int | None] = {}
        for item in eval_items:
            request = Request(scoring_messages(aspect, item, cot), config.scoring_temperature, config.score_max_tokens, 0)
            scores[item.uuid] = parse_score(llm.complete(request), spec)
        corr = correlate([scores[item.uuid] for item in eval_items], gold, groups, config.selection_level)
        candidate = Candidate(
            id=f"{origin[0]}{len(candidates):03d}",
            origin=origin,
            parent=parent,
            cot=cot,
            shots=[item.uuid for item in shots],
            temperature=temperature,
            spearman=corr.spearman,
            kendall=corr.kendall,
            n_valid=corr.n,
            scores=scores,
        )
        candidates.append(candidate)
        if progress:
            progress(candidate)
        return candidate

    drafts = []
    for size in config.shot_sizes:
        for _ in range(config.n_trials):
            shots = rng.sample(usable, size)
            temperature = rng.uniform(*config.draft_temperature)
            request = Request(drafting_messages(aspect, shots), temperature, config.draft_max_tokens, rng.randrange(2**31))
            drafts.append(evaluate(llm.complete(request).strip(), "draft", None, shots, temperature))

    for parent in sorted(drafts, key=lambda c: c.fitness, reverse=True)[: config.top_k]:
        misaligned = most_misaligned(parent, eval_items, aspect, config.misaligned_pool)
        if not misaligned:
            continue
        for _ in range(config.n_trials):
            chosen = rng.sample(misaligned, min(config.misaligned_shots, len(misaligned)))
            request = Request(
                refinement_messages(aspect, parent.cot, chosen),
                config.refine_temperature,
                config.draft_max_tokens,
                rng.randrange(2**31),
            )
            evaluate(llm.complete(request).strip(), "refined", parent.id, [item for item, _ in chosen], config.refine_temperature)

    best = max(candidates, key=lambda c: c.fitness)
    return CalibrationResult(
        aspect=aspect,
        model=llm.model,
        config=config,
        data_fingerprint=data_fingerprint,
        eval_uuids=[item.uuid for item in eval_items],
        candidates=candidates,
        best_id=best.id,
    )


def load_best_steps(run_dir: Path | str) -> dict[Aspect, str]:
    steps = {}
    for path in sorted(Path(run_dir, "calibration").glob("*.json")):
        result = CalibrationResult.load(path)
        steps[result.aspect] = result.best.cot
    if not steps:
        raise FileNotFoundError(f"no calibration results in {run_dir}/calibration")
    return steps
