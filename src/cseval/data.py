from __future__ import annotations

import csv
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .aspects import ALL_ASPECTS, SPECS, Aspect

HUB_REPO = "Aswini123/CSEval"
DEFAULT_DATA_DIR = Path("data/CSEval-hf")


@dataclass(frozen=True)
class Item:
    uuid: str
    hs_id: str
    hatespeech: str
    counterspeech: str
    reference: str
    cs_type: str
    annotator: str
    gold: Mapping[Aspect, float | None]
    in_dev: bool


@dataclass(frozen=True)
class Dataset:
    items: tuple[Item, ...]
    rows_without_prediction: int
    fingerprint: str
    source: str

    def calibration_set(self) -> list[Item]:
        return [item for item in self.items if item.in_dev]

    def test_set(self) -> list[Item]:
        return [item for item in self.items if not item.in_dev]

    def by_uuid(self) -> dict[str, Item]:
        return {item.uuid: item for item in self.items}


def hs_id(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:12]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _fingerprint(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_dataset(data_dir: Path | str = DEFAULT_DATA_DIR) -> Dataset:
    data_dir = Path(data_dir)
    main_path, dev_path = data_dir / "cs-eval.csv", data_dir / "dev.csv"
    if not main_path.exists():
        raise FileNotFoundError(f"{main_path} not found; run `cseval download` or pass --data-dir")
    rows = _read_csv(main_path)
    dev_uuids = {row["uuid"] for row in _read_csv(dev_path)}
    missing = dev_uuids - {row["uuid"] for row in rows}
    if missing:
        raise ValueError(f"{len(missing)} uuids in dev.csv are not in cs-eval.csv")

    items, without_prediction = [], 0
    for row in rows:
        if not row["predicted_counterspeech"].strip():
            without_prediction += 1
            continue
        gold: dict[Aspect, float | None] = {}
        for aspect in ALL_ASPECTS:
            raw = row[SPECS[aspect].column].strip()
            gold[aspect] = float(raw) if raw else None
        items.append(
            Item(
                uuid=row["uuid"],
                hs_id=hs_id(row["hatespeech"]),
                hatespeech=row["hatespeech"].strip(),
                counterspeech=row["predicted_counterspeech"].strip(),
                reference=row["gold_counterspeech"].strip(),
                cs_type=row["csType"],
                annotator=row["annotator"],
                gold=gold,
                in_dev=row["uuid"] in dev_uuids,
            )
        )
    return Dataset(
        items=tuple(items),
        rows_without_prediction=without_prediction,
        fingerprint=_fingerprint([main_path, dev_path]),
        source=str(data_dir),
    )


def download(data_dir: Path | str = DEFAULT_DATA_DIR) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=HUB_REPO, repo_type="dataset", local_dir=str(data_dir)))
