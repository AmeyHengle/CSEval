from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from .prompts import Messages


@dataclass(frozen=True)
class Request:
    messages: Messages
    temperature: float
    max_tokens: int
    seed: int | None = None

    def key(self, model: str) -> str:
        payload = json.dumps(
            {
                "model": model,
                "messages": [list(message) for message in self.messages],
                "temperature": round(self.temperature, 6),
                "max_tokens": self.max_tokens,
                "seed": self.seed,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LLM(Protocol):
    model: str

    def complete(self, request: Request) -> str: ...


PROVIDERS: dict[str, dict[str, str | None]] = {
    "openai": {"base_url": None, "api_key_env": "OPENAI_API_KEY"},
    "together": {"base_url": "https://api.together.xyz/v1", "api_key_env": "TOGETHER_API_KEY"},
    "vllm": {"base_url": "http://localhost:8000/v1", "api_key_env": None},
}


class OpenAICompatibleLLM:
    def __init__(
        self,
        model: str,
        provider: str = "openai",
        base_url: str | None = None,
        api_key_env: str | None = None,
        send_seed: bool = True,
        timeout: float = 60.0,
        max_retries: int = 5,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("install the `llm` extra: uv sync --extra llm") from exc
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}; expected one of {sorted(PROVIDERS)}")
        defaults = PROVIDERS[provider]
        key_env = api_key_env or defaults["api_key_env"]
        api_key = os.environ.get(key_env) if key_env else "not-needed"
        if not api_key:
            raise RuntimeError(f"environment variable {key_env} is not set")
        self.model = model
        self.provider = provider
        self.send_seed = send_seed
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url or defaults["base_url"],
            timeout=timeout,
            max_retries=max_retries,
        )

    def complete(self, request: Request) -> str:
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": role, "content": content} for role, content in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if self.send_seed and request.seed is not None:
            kwargs["seed"] = request.seed
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


class CachedLLM:
    def __init__(self, llm: LLM, path: Path | str) -> None:
        self.llm = llm
        self.model = llm.model
        self.hits = self.misses = 0
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, model TEXT, request TEXT, response TEXT, created TEXT)"
        )

    def complete(self, request: Request) -> str:
        key = request.key(self.model)
        row = self._db.execute("SELECT response FROM calls WHERE key = ?", (key,)).fetchone()
        if row is not None:
            self.hits += 1
            return row[0]
        self.misses += 1
        response = self.llm.complete(request)
        self._db.execute(
            "INSERT OR REPLACE INTO calls VALUES (?, ?, ?, ?, ?)",
            (
                key,
                self.model,
                json.dumps([list(m) for m in request.messages], ensure_ascii=False),
                response,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._db.commit()
        return response


@dataclass
class FakeLLM:

    respond: Callable[[Request], str]
    model: str = "fake-llm"
    calls: list[Request] = field(default_factory=list)

    def complete(self, request: Request) -> str:
        self.calls.append(request)
        return self.respond(request)
