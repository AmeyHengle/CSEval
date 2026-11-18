from __future__ import annotations

import json
import re

from .aspects import AspectSpec

_JSON_OBJECT = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_score(text: str | None, spec: AspectSpec) -> int | None:
    for match in _JSON_OBJECT.finditer(text or ""):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "score" not in obj:
            continue
        value = obj["score"]
        if isinstance(value, bool):
            return None
        try:
            number = float(str(value).strip())
        except ValueError:
            return None
        if number.is_integer() and spec.in_scale(number):
            return int(number)
        return None
    return None
