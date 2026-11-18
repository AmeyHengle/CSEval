import itertools

from cseval.aspects import SPECS, Aspect
from cseval.calibration import CalibrationConfig, CalibrationResult, calibrate, most_misaligned
from cseval.llm import FakeLLM, Request

from .conftest import make_item

ITEMS = [
    make_item(f"u{i:02d}", hatespeech=f"Hate speech {i}.", relevance=float(1 + i % 5), coherence=float(5 - i % 5))
    for i in range(20)
]
GOLD = {item.counterspeech: item.gold for item in ITEMS}
FLIPPED = {ITEMS[0].counterspeech, ITEMS[4].counterspeech}


def scripted(aspect: Aspect):
    drafts, refinements = itertools.count(), itertools.count()

    def respond(request: Request) -> str:
        first = request.messages[0][1]
        if first.startswith("## Objective:"):
            return f"draft-{next(drafts)}"
        if first.startswith("Please refine and improve"):
            parent = first.split("Old criteria: ")[1].split("\n")[0]
            return f"refined-of-{parent}-{next(refinements)}"
        steps = first.split("## Evaluation Steps:\n")[1]
        counterspeech = request.messages[1][1].split("Counterspeech: ")[1].split("\n")[0]
        gold = int(GOLD[counterspeech][aspect])
        if steps.startswith("refined-of-draft-0"):
            return f'{{"score": {gold}}}'
        if steps == "draft-0":
            return f'{{"score": {6 - gold if counterspeech in FLIPPED else gold}}}'
        if steps == "draft-1":
            return f'{{"score": {6 - gold}}}'
        return '{"score": 3}'

    return respond


CONFIG = CalibrationConfig(n_trials=4, shot_sizes=(3,), top_k=2, misaligned_pool=5, misaligned_shots=3, seed=7)


def run(aspect=Aspect.RELEVANCE):
    llm = FakeLLM(scripted(aspect))
    return llm, calibrate(llm, ITEMS, aspect, CONFIG)


def test_pool_has_drafts_and_refinements_of_the_top_k():
    _, result = run()
    drafts = [c for c in result.candidates if c.origin == "draft"]
    refined = [c for c in result.candidates if c.origin == "refined"]
    assert len(drafts) == 4 and len(refined) == 2 * 4
    assert {c.parent for c in refined} == {drafts[0].id, drafts[1].id}


def test_best_candidate_is_selected_by_spearman_over_the_whole_pool():
    _, result = run()
    assert result.best.origin == "refined" and result.best.cot.startswith("refined-of-draft-0")
    assert result.best.spearman == 1.0 and result.best.n_valid == len(ITEMS)


def test_refinement_examples_come_from_the_most_misaligned_rows():
    llm, result = run()
    parent = next(c for c in result.candidates if c.cot == "draft-0")
    worst = most_misaligned(parent, ITEMS, Aspect.RELEVANCE, 2)
    assert {item.counterspeech for item, _ in worst} == FLIPPED
    pool = {item.counterspeech for item, _ in most_misaligned(parent, ITEMS, Aspect.RELEVANCE, CONFIG.misaligned_pool)}
    for request in llm.calls:
        text = request.messages[0][1]
        if text.startswith("Please refine") and "Old criteria: draft-0\n" in text:
            shown = {line.split(": ", 1)[1] for line in text.splitlines() if line.startswith("Counterspeech: ")}
            assert len(shown) == 3 and shown <= pool


def test_drafting_uses_the_calibrated_aspect():
    llm, result = run(Aspect.COHERENCE)
    drafting = [r.messages[0][1] for r in llm.calls if r.messages[0][1].startswith("## Objective:")]
    assert drafting and all("Expert Rating (Coherence)" in text for text in drafting)
    assert all(SPECS[Aspect.COHERENCE].criteria in text for text in drafting)
    assert result.best.spearman == 1.0


def test_draft_temperatures_stay_in_range_and_runs_are_reproducible(tmp_path):
    llm, first = run()
    temps = [r.temperature for r in llm.calls if r.messages[0][1].startswith("## Objective:")]
    assert all(0.0 <= t <= 0.5 for t in temps)
    _, second = run()
    assert [(c.cot, c.shots) for c in first.candidates] == [(c.cot, c.shots) for c in second.candidates]
    path = tmp_path / "relevance.json"
    first.save(path)
    loaded = CalibrationResult.load(path)
    assert loaded.best.cot == first.best.cot and loaded.config == first.config


def test_call_budget_matches_the_algorithm():
    llm, _ = run()
    assert len(llm.calls) == CONFIG.calls(len(ITEMS))
