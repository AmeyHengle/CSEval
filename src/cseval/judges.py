from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from .aspects import SPECS, Aspect
from .data import Item
from .llm import LLM, Request
from .parsing import parse_score
from .prompts import geval_steps_messages, scoring_messages

RETRY_MESSAGE = 'Your reply could not be read. Respond with a JSON object only, in the form {"score": <integer>}.'


@dataclass(frozen=True)
class Prediction:
    uuid: str
    aspect: Aspect
    score: int | None
    attempts: int
    response: str


@dataclass
class PromptJudge:
    llm: LLM
    name: str
    steps: Mapping[Aspect, str | None]
    temperature: float = 0.0
    max_tokens: int = 64
    retries: int = 1
    seed: int | None = 0

    def score(self, item: Item, aspect: Aspect) -> Prediction:
        spec = SPECS[aspect]
        messages = scoring_messages(aspect, item, self.steps.get(aspect))
        text = ""
        for attempt in range(1, self.retries + 2):
            text = self.llm.complete(Request(messages, self.temperature, self.max_tokens, self.seed))
            score = parse_score(text, spec)
            if score is not None:
                return Prediction(item.uuid, aspect, score, attempt, text)
            messages = messages + (("assistant", text), ("user", RETRY_MESSAGE))
        return Prediction(item.uuid, aspect, None, self.retries + 1, text)


def zero_shot_judge(llm: LLM, **options) -> PromptJudge:
    return PromptJudge(llm, "zero-shot", {aspect: None for aspect in Aspect}, **options)


def geval_judge(llm: LLM, aspects: Iterable[Aspect], steps_temperature: float = 0.0, **options) -> PromptJudge:
    steps = {
        aspect: llm.complete(Request(geval_steps_messages(aspect), steps_temperature, 512, 0)).strip()
        for aspect in aspects
    }
    return PromptJudge(llm, "g-eval", steps, **options)


def auto_cseval_judge(llm: LLM, steps: Mapping[Aspect, str], **options) -> PromptJudge:
    return PromptJudge(llm, "auto-cseval", dict(steps), **options)


def score_items(
    judge: PromptJudge,
    items: Sequence[Item],
    aspects: Iterable[Aspect],
    progress: Callable[[int, int], None] | None = None,
) -> list[Prediction]:
    aspects = list(aspects)
    total, done, predictions = len(items) * len(aspects), 0, []
    for item in items:
        for aspect in aspects:
            predictions.append(judge.score(item, aspect))
            done += 1
            if progress:
                progress(done, total)
    return predictions
