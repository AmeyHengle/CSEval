from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .llm import CachedLLM, OpenAICompatibleLLM


def load_config(path: Path | str) -> dict:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    name: str
    base_url: str | None = None
    api_key_env: str | None = None
    send_seed: bool = True
    cache: str = ".cache/llm.sqlite"

    @classmethod
    def from_dict(cls, values: Mapping) -> ModelConfig:
        return cls(**values)


def build_llm(config: ModelConfig) -> CachedLLM:
    client = OpenAICompatibleLLM(
        config.name,
        provider=config.provider,
        base_url=config.base_url,
        api_key_env=config.api_key_env,
        send_seed=config.send_seed,
    )
    return CachedLLM(client, config.cache)
