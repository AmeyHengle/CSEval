from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

Level = Literal["dataset", "sample"]
NAN = float("nan")


@dataclass(frozen=True)
class Correlation:
    spearman: float
    kendall: float
    n: int
    groups: int | None = None


def _valid_pairs(pred: Sequence[float | None], gold: Sequence[float | None]) -> tuple[np.ndarray, np.ndarray]:
    pairs = [(p, g) for p, g in zip(pred, gold, strict=True) if p is not None and g is not None]
    if not pairs:
        return np.array([]), np.array([])
    x, y = zip(*pairs, strict=True)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _rank_correlations(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 2 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return NAN, NAN
    return float(stats.spearmanr(x, y).statistic), float(stats.kendalltau(x, y, variant="b").statistic)


def dataset_level(pred: Sequence[float | None], gold: Sequence[float | None]) -> Correlation:
    x, y = _valid_pairs(pred, gold)
    rho, tau = _rank_correlations(x, y)
    return Correlation(rho, tau, len(x))


def sample_level(
    pred: Sequence[float | None],
    gold: Sequence[float | None],
    groups: Sequence[Hashable],
) -> Correlation:
    by_group: dict[Hashable, list[tuple[float | None, float | None]]] = defaultdict(list)
    for p, g, key in zip(pred, gold, groups, strict=True):
        by_group[key].append((p, g))
    rhos, taus, n = [], [], 0
    for rows in by_group.values():
        x, y = _valid_pairs([p for p, _ in rows], [g for _, g in rows])
        rho, tau = _rank_correlations(x, y)
        if math.isnan(rho):
            continue
        rhos.append(rho)
        taus.append(tau)
        n += len(x)
    if not rhos:
        return Correlation(NAN, NAN, 0, 0)
    return Correlation(float(np.mean(rhos)), float(np.nanmean(taus)), n, len(rhos))


def correlate(
    pred: Sequence[float | None],
    gold: Sequence[float | None],
    groups: Sequence[Hashable],
    level: Level,
) -> Correlation:
    return dataset_level(pred, gold) if level == "dataset" else sample_level(pred, gold, groups)


def unified_score(relevance: float, aggressiveness: float, coherence: float, suitableness: float) -> float:
    return (relevance + (6 - aggressiveness) + coherence) / 4 + (((suitableness - 1) / 2) * 4 + 1) / 4
