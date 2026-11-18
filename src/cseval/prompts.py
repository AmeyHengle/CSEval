from __future__ import annotations

from collections.abc import Sequence
from importlib.resources import files
from string import Template

from .aspects import SPECS, Aspect
from .data import Item

Messages = tuple[tuple[str, str], ...]


def _render(template: str, **values: object) -> str:
    text = files("cseval").joinpath("templates", template).read_text(encoding="utf-8")
    return Template(text).substitute(**values).strip()


def _with_article(phrase: str) -> str:
    return f"{'an' if phrase[0].lower() in 'aeiou' else 'a'} {phrase}"


def drafting_exemplar(item: Item, aspect: Aspect) -> str:
    name = SPECS[aspect].name
    return (
        f"Hatespeech: {item.hatespeech}\n"
        f"Counterspeech: {item.counterspeech}\n"
        f"Expert Rating ({name}): {item.gold[aspect]:.1f}"
    )


def refinement_exemplar(item: Item, aspect: Aspect, predicted: int) -> str:
    name = SPECS[aspect].name
    return f"{drafting_exemplar(item, aspect)}\nLLM Rating ({name}): {predicted}"


def drafting_messages(aspect: Aspect, shots: Sequence[Item]) -> Messages:
    spec = SPECS[aspect]
    text = _render(
        "drafting.txt",
        aspect=spec.name,
        aspect_with_scale=_with_article(f"{spec.name} score ({spec.scale})"),
        criteria=spec.criteria,
        examples="\n\n".join(drafting_exemplar(item, aspect) for item in shots),
    )
    return (("user", text),)


def refinement_messages(aspect: Aspect, previous_cot: str, misaligned: Sequence[tuple[Item, int]]) -> Messages:
    text = _render(
        "refinement.txt",
        aspect=SPECS[aspect].name,
        previous_cot=previous_cot.strip(),
        examples="\n\n".join(refinement_exemplar(item, aspect, pred) for item, pred in misaligned),
    )
    return (("user", text),)


def scoring_messages(aspect: Aspect, item: Item, steps: str | None) -> Messages:
    spec = SPECS[aspect]
    block = f"\n\n## Evaluation Steps:\n{steps.strip()}" if steps else ""
    system = _render("scoring_system.txt", criteria=spec.criteria, evaluation_steps=block)
    user = _render(
        "scoring_user.txt",
        hatespeech=item.hatespeech,
        counterspeech=item.counterspeech,
        low=spec.low,
        high=spec.high,
    )
    return (("system", system), ("user", user))


def geval_steps_messages(aspect: Aspect) -> Messages:
    return (("user", _render("geval_steps.txt", criteria=SPECS[aspect].criteria)),)
