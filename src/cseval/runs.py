from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .data import Dataset


def create_run(root: Path | str, name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = Path(root) / f"{stamp}-{name}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def manifest(command: str, config: Mapping, dataset: Dataset, model: str | None = None) -> dict:
    status = _git("status", "--porcelain")
    return {
        "command": command,
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "python": sys.version.split()[0],
        "packages": {name: _version(name) for name in ("cseval", "numpy", "scipy", "openai")},
        "data_source": dataset.source,
        "data_fingerprint": dataset.fingerprint,
        "model": model,
        "config": dict(config),
    }


def write_json(path: Path | str, obj: object) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: Path | str, records: Iterable[Mapping]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path | str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
