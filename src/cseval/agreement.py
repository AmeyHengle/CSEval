from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .aspects import ALL_ASPECTS, SPECS, Aspect


def krippendorff_alpha_interval(units: Sequence[Sequence[float | None]]) -> float:
    pairable = [np.asarray([v for v in unit if v is not None], dtype=float) for unit in units]
    pairable = [values for values in pairable if len(values) >= 2]
    if not pairable:
        return float("nan")
    n = sum(len(values) for values in pairable)
    observed = sum(
        2 * len(values) * ((values - values.mean()) ** 2).sum() / (len(values) - 1) for values in pairable
    ) / n
    every = np.concatenate(pairable)
    expected = 2 * ((every - every.mean()) ** 2).sum() / (n - 1)
    return float(1 - observed / expected) if expected > 0 else float("nan")


def load_ratings(path: Path | str) -> dict[Aspect, list[list[float | None]]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ratings: dict[Aspect, list[list[float | None]]] = {aspect: [] for aspect in ALL_ASPECTS}
    for row in rows:
        for aspect in ALL_ASPECTS:
            column = SPECS[aspect].column
            unit = []
            for k in (1, 2, 3):
                raw = row[f"{column}_annotator{k}"].strip()
                unit.append(float(raw) if raw else None)
            ratings[aspect].append(unit)
    return ratings
