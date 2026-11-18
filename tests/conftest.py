from __future__ import annotations

from pathlib import Path

import pytest

from cseval.aspects import Aspect
from cseval.data import Dataset, Item, hs_id, load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "CSEval-hf"


@pytest.fixture(scope="session")
def dataset() -> Dataset:
    if not (DATA_DIR / "cs-eval.csv").exists():
        pytest.skip("data/CSEval-hf not found")
    return load_dataset(DATA_DIR)


def make_item(uuid: str, hatespeech: str = "Women are too emotional to be leaders.", **gold: float) -> Item:
    scores = {aspect: None for aspect in Aspect}
    scores.update({Aspect(name): value for name, value in gold.items()})
    return Item(
        uuid=uuid,
        hs_id=hs_id(hatespeech),
        hatespeech=hatespeech,
        counterspeech=f"Counterspeech {uuid}.",
        reference="A reference counterspeech.",
        cs_type="Informative",
        annotator="",
        gold=scores,
        in_dev=True,
    )
