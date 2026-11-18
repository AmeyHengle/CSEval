import math

import pytest
from scipy import stats

from cseval.agreement import krippendorff_alpha_interval, load_ratings
from cseval.evaluation import dataset_level, sample_level, unified_score

from .conftest import DATA_DIR


def test_dataset_level_matches_scipy_and_skips_missing():
    pred = [1, 2, None, 4, 5, 3, 3]
    gold = [1, 3, 2, 4, 4, None, 2]
    corr = dataset_level(pred, gold)
    x, y = [1, 2, 4, 5, 3], [1, 3, 4, 4, 2]
    assert corr.n == 5
    assert corr.spearman == pytest.approx(stats.spearmanr(x, y).statistic)
    assert corr.kendall == pytest.approx(stats.kendalltau(x, y, variant="b").statistic)


def test_constant_scores_have_no_correlation():
    assert math.isnan(dataset_level([3, 3, 3], [1, 2, 3]).spearman)


def test_sample_level_averages_usable_groups_only():
    pred = [1, 2, 3, 5, 4, 2, 2]
    gold = [1, 2, 3, 1, 2, 4, 5]
    groups = ["a", "a", "a", "b", "b", "c", "c"]
    corr = sample_level(pred, gold, groups)
    assert corr.groups == 2
    assert corr.spearman == pytest.approx(0.0)
    assert corr.n == 5


def test_unified_score_range():
    assert unified_score(5, 1, 5, 3) == 5.0
    assert unified_score(1, 5, 1, 1) == 1.0


def test_krippendorff_alpha_known_values():
    assert krippendorff_alpha_interval([[1, 1, 1], [3, 3, 3], [5, 5, None]]) == pytest.approx(1.0)
    a = [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None]
    b = [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3]
    c = [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None]
    d = [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None]
    units = [list(unit) for unit in zip(a, b, c, d)]
    assert krippendorff_alpha_interval(units) == pytest.approx(0.849, abs=1e-3)


def test_agreement_on_the_release(dataset):
    ratings = load_ratings(DATA_DIR / "inter-annotator.csv")
    for units in ratings.values():
        assert len(units) == 1500
        assert 0 < krippendorff_alpha_interval(units) < 1
