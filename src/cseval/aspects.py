from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Aspect(StrEnum):
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    AGGRESSIVENESS = "aggressiveness"
    SUITABLENESS = "suitableness"


@dataclass(frozen=True)
class AspectSpec:
    aspect: Aspect
    name: str
    low: int
    high: int
    criteria: str

    @property
    def column(self) -> str:
        return f"{self.aspect.value}_score"

    @property
    def scale(self) -> str:
        return f"{self.low}-{self.high}"

    def in_scale(self, value: float) -> bool:
        return self.low <= value <= self.high


SPECS: dict[Aspect, AspectSpec] = {
    Aspect.RELEVANCE: AspectSpec(
        Aspect.RELEVANCE,
        "Relevance",
        1,
        5,
        "Contextual Relevance (1-5) - This dimension evaluates whether the counterspeech addresses "
        "the central topic, theme subject of the given hate speech.",
    ),
    Aspect.COHERENCE: AspectSpec(
        Aspect.COHERENCE,
        "Coherence",
        1,
        5,
        "Argument Coherence (1-5) - This metric assesses how logically and smoothly the ideas or "
        "arguments within the counterspeech connect and flow. A coherent counterspeech will present "
        "its arguments in an organized manner, making it easy for the reader to follow and understand "
        "the counter-narrative being presented.",
    ),
    Aspect.AGGRESSIVENESS: AspectSpec(
        Aspect.AGGRESSIVENESS,
        "Aggressiveness",
        1,
        5,
        "Aggressiveness (1-5) - This metric assesses the degree of confrontational or inflammatory "
        "content in the counterspeech. It considers factors such as abusive language, intensity of "
        "disagreement, tone, engagement style, and the presence of personal attacks, with higher "
        "scores indicating greater aggression.",
    ),
    Aspect.SUITABLENESS: AspectSpec(
        Aspect.SUITABLENESS,
        "Suitableness",
        1,
        3,
        "Suitableness (1-3) - This metric measures the likelihood of an annotator choosing a given "
        "counterspeech for direct use (without editing) in a real scenario against online hate speech. "
        "This assessment considers the counterspeech's suitability, appropriateness, and potential "
        "impact on a reader in a real-world context.",
    ),
}

ALL_ASPECTS: tuple[Aspect, ...] = tuple(Aspect)
