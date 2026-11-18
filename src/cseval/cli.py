from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

from .agreement import krippendorff_alpha_interval, load_ratings
from .aspects import ALL_ASPECTS, Aspect
from .calibration import CalibrationConfig, Candidate, calibrate, load_best_steps
from .config import ModelConfig, build_llm, load_config
from .data import DEFAULT_DATA_DIR, download, load_dataset
from .evaluation import Level, correlate
from .judges import auto_cseval_judge, geval_judge, score_items, zero_shot_judge
from .runs import create_run, manifest, read_json, read_jsonl, write_json, write_jsonl

TABLE_ORDER = (Aspect.RELEVANCE, Aspect.AGGRESSIVENESS, Aspect.COHERENCE, Aspect.SUITABLENESS)


def _confirm(calls: int, model: str, assume_yes: bool) -> None:
    print(f"Up to {calls:,} calls to {model} (cached calls cost nothing).")
    if assume_yes:
        return
    if not sys.stdin.isatty():
        raise SystemExit("not starting without --yes in a non-interactive shell")
    if input("Proceed? [y/N] ").strip().lower() != "y":
        raise SystemExit("aborted")


def _progress(done: int, total: int) -> None:
    if done == total or done % max(1, total // 20) == 0:
        print(f"\r  {done:,}/{total:,}", end="\n" if done == total else "", file=sys.stderr, flush=True)


def _fmt(value: float) -> str:
    return "—" if math.isnan(value) else f"{value:.3f}"


def _mean(values: list[float]) -> float:
    valid = [v for v in values if not math.isnan(v)]
    return sum(valid) / len(valid) if valid else math.nan


def cmd_data(args: argparse.Namespace) -> None:
    ds = load_dataset(args.data_dir)
    print(f"Source: {ds.source} (fingerprint {ds.fingerprint[:12]})")
    print(f"Rows in cs-eval.csv: {len(ds.items) + ds.rows_without_prediction:,}")
    print(f"  with predicted_counterspeech: {len(ds.items):,}")
    print(f"  without predicted_counterspeech: {ds.rows_without_prediction:,}")
    print(f"Hate speeches with predicted_counterspeech: {len({item.hs_id for item in ds.items}):,}")
    print(f"Calibration set (dev.csv): {len(ds.calibration_set()):,} rows")
    print(f"Test set (rows not in dev.csv): {len(ds.test_set()):,} rows")


def cmd_agreement(args: argparse.Namespace) -> None:
    ratings = load_ratings(Path(args.data_dir) / "inter-annotator.csv")
    alphas = {aspect: krippendorff_alpha_interval(units) for aspect, units in ratings.items()}
    print(f"Krippendorff's alpha (interval), {len(ratings[Aspect.RELEVANCE]):,} rows x 3 annotators")
    for aspect in TABLE_ORDER:
        print(f"  {aspect.value:15s} {alphas[aspect]:.3f}")
    print(f"  {'mean':15s} {sum(alphas.values()) / len(alphas):.3f}")


def cmd_download(args: argparse.Namespace) -> None:
    print(f"Downloaded to {download(args.data_dir)}")


def cmd_calibrate(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    section = dict(cfg["calibration"])
    configured = section.pop("aspects", [aspect.value for aspect in ALL_ASPECTS])
    aspects = [Aspect(a) for a in (args.aspect or configured)]
    config = CalibrationConfig.from_dict(section)
    ds = load_dataset(cfg["data"]["dir"])
    calibration_items = ds.calibration_set()
    calls = 0
    for aspect in aspects:
        usable = sum(item.gold[aspect] is not None for item in calibration_items)
        calls += config.calls(min(usable, config.eval_size or usable))
    model = ModelConfig.from_dict(cfg["model"])
    _confirm(calls, model.name, args.yes)
    llm = build_llm(model)
    run_dir = create_run(cfg.get("run", {}).get("dir", "runs"), "calibrate")
    write_json(run_dir / "manifest.json", manifest("calibrate", cfg, ds, model.name))
    (run_dir / "calibration").mkdir()

    def report(candidate: Candidate) -> None:
        print(f"  {candidate.id} {candidate.origin:7s} spearman={candidate.spearman:.3f} valid={candidate.n_valid}", file=sys.stderr)

    for aspect in aspects:
        print(f"Calibrating {aspect.value}", file=sys.stderr)
        result = calibrate(llm, calibration_items, aspect, config, ds.fingerprint, report)
        result.save(run_dir / "calibration" / f"{aspect.value}.json")
        best = result.best
        print(f"{aspect.value}: best {best.id} ({best.origin}) spearman={best.spearman:.3f} on {best.n_valid} rows")
    print(f"Run: {run_dir} (cache hits {llm.hits:,}, new calls {llm.misses:,})")


def cmd_score(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    data, judge_cfg = cfg["data"], cfg["judge"]
    ds = load_dataset(data["dir"])
    items = ds.calibration_set() if data.get("split", "test") == "calibration" else ds.test_set()
    if args.limit:
        items = items[: args.limit]
    aspects = [Aspect(a) for a in judge_cfg.get("aspects", [aspect.value for aspect in ALL_ASPECTS])]
    kind = judge_cfg["kind"]
    options = {key: judge_cfg[key] for key in ("temperature", "max_tokens", "retries") if key in judge_cfg}
    model = ModelConfig.from_dict(cfg["model"])
    _confirm(len(items) * len(aspects) * (1 + options.get("retries", 1)), model.name, args.yes)
    llm = build_llm(model)
    if kind == "zero-shot":
        judge = zero_shot_judge(llm, **options)
    elif kind == "g-eval":
        judge = geval_judge(llm, aspects, **options)
    elif kind == "auto-cseval":
        steps = load_best_steps(judge_cfg["calibration_run"])
        missing = [aspect.value for aspect in aspects if aspect not in steps]
        if missing:
            raise SystemExit(f"calibration run has no result for {missing}")
        judge = auto_cseval_judge(llm, steps, **options)
    else:
        raise SystemExit(f"unknown judge kind {kind!r}")
    run_dir = create_run(cfg.get("run", {}).get("dir", "runs"), f"score-{kind}")
    write_json(run_dir / "manifest.json", manifest("score", {**cfg, "limit": args.limit}, ds, model.name))
    write_json(run_dir / "steps.json", {aspect.value: judge.steps.get(aspect) for aspect in aspects})
    predictions = score_items(judge, items, aspects, _progress)
    write_jsonl(
        run_dir / "predictions.jsonl",
        (
            {"uuid": p.uuid, "aspect": p.aspect.value, "score": p.score, "attempts": p.attempts, "response": p.response}
            for p in predictions
        ),
    )
    unparsed = Counter(p.aspect.value for p in predictions if p.score is None)
    print(f"Scored {len(items):,} rows x {len(aspects)} aspects; unparsed: {dict(unparsed) or 'none'}")
    print(f"Run: {run_dir} (cache hits {llm.hits:,}, new calls {llm.misses:,})")


def cmd_evaluate(args: argparse.Namespace) -> None:
    runs = [Path(run) for run in args.runs]
    manifests = [read_json(run / "manifest.json") for run in runs]
    ds = load_dataset(args.data_dir or manifests[0]["config"]["data"]["dir"])
    for run, man in zip(runs, manifests, strict=True):
        if man["data_fingerprint"] != ds.fingerprint:
            print(f"warning: {run} was scored on different data ({man['data_fingerprint'][:12]})", file=sys.stderr)
    by_uuid = ds.by_uuid()
    levels: list[Level] = ["dataset", "sample"] if args.level == "both" else [args.level]
    results: dict[str, dict] = {}
    table: dict[str, list[str]] = {level: [] for level in levels}
    used = []
    for run, man in zip(runs, manifests, strict=True):
        label = f"{man['config']['judge']['kind']} · {man['model']}"
        preds: dict[Aspect, dict[str, int | None]] = {}
        for record in read_jsonl(run / "predictions.jsonl"):
            preds.setdefault(Aspect(record["aspect"]), {})[record["uuid"]] = record["score"]
        results[label] = {}
        counts = []
        for aspect in TABLE_ORDER:
            if aspect not in preds:
                continue
            items = [by_uuid[uuid] for uuid in preds[aspect] if uuid in by_uuid]
            pred = [preds[aspect][item.uuid] for item in items]
            gold = [item.gold[aspect] for item in items]
            groups = [item.hs_id for item in items]
            results[label][aspect.value] = {level: vars(correlate(pred, gold, groups, level)) for level in levels}
            counts.append(f"{aspect.value}: n={len(items):,}, unparsed={sum(p is None for p in pred)}")
        used.append(f"- {label}: " + "; ".join(counts))
        for level in levels:
            cells, rhos, taus = [], [], []
            for aspect in TABLE_ORDER:
                corr = results[label].get(aspect.value, {}).get(level)
                if corr is None:
                    cells += ["", ""]
                    continue
                cells += [_fmt(corr["spearman"]), _fmt(corr["kendall"])]
                rhos.append(corr["spearman"])
                taus.append(corr["kendall"])
            cells += [_fmt(_mean(rhos)), _fmt(_mean(taus))]
            table[level].append(f"| {label} | " + " | ".join(cells) + " |")
    header = "| Judge | " + " | ".join(f"{a.value[:3].title()} ρ | {a.value[:3].title()} τ" for a in TABLE_ORDER) + " | AVG ρ | AVG τ |"
    for level in levels:
        print(f"\n### {level.title()}-level correlation with gold\n")
        print(header)
        print("|" + "---|" * (2 * len(TABLE_ORDER) + 3))
        print("\n".join(table[level]))
    print("\nRows used:\n" + "\n".join(used))
    if args.out:
        write_json(args.out, results)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cseval")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("data", help="summarise the dataset and its splits")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.set_defaults(func=cmd_data)

    p = sub.add_parser("agreement", help="Krippendorff's alpha on inter-annotator.csv")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.set_defaults(func=cmd_agreement)

    p = sub.add_parser("download", help="download the dataset from the Hugging Face Hub")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("calibrate", help="Auto-CSEval calibration on the calibration set")
    p.add_argument("config")
    p.add_argument("--aspect", action="append", choices=[aspect.value for aspect in ALL_ASPECTS])
    p.add_argument("--yes", action="store_true", help="do not ask before calling the API")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("score", help="score a split with one judge")
    p.add_argument("config")
    p.add_argument("--limit", type=int, help="score only the first N rows")
    p.add_argument("--yes", action="store_true", help="do not ask before calling the API")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("evaluate", help="correlate score runs with the gold scores")
    p.add_argument("runs", nargs="+")
    p.add_argument("--level", choices=["dataset", "sample", "both"], default="both")
    p.add_argument("--data-dir")
    p.add_argument("--out", help="also write the results as JSON")
    p.set_defaults(func=cmd_evaluate)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
