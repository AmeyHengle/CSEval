import csv

from cseval.aspects import SPECS

from .conftest import DATA_DIR


def _rows(name):
    with (DATA_DIR / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_release_counts(dataset):
    rows = _rows("cs-eval.csv")
    assert len(rows) == 7926
    assert len({r["hatespeech"] for r in rows}) == 2223
    assert len({r["gold_counterspeech"] for r in rows}) == 4318


def test_items_are_rows_with_a_predicted_counterspeech(dataset):
    assert len(dataset.items) == 6526
    assert dataset.rows_without_prediction == 1400
    assert all(item.counterspeech for item in dataset.items)


def test_calibration_and_test_sets_are_disjoint(dataset):
    calibration, test = dataset.calibration_set(), dataset.test_set()
    assert len(calibration) == 500
    assert len(test) == 6026
    assert {item.uuid for item in calibration}.isdisjoint(item.uuid for item in test)


def test_dev_file_agrees_with_main_file(dataset):
    main = {row["uuid"]: row for row in _rows("cs-eval.csv")}
    for row in _rows("dev.csv"):
        for spec in SPECS.values():
            assert row[spec.column] == main[row["uuid"]][spec.column]
