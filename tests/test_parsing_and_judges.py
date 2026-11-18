from cseval.aspects import SPECS, Aspect
from cseval.judges import RETRY_MESSAGE, auto_cseval_judge, zero_shot_judge
from cseval.llm import FakeLLM
from cseval.parsing import parse_score

from .conftest import make_item

REL = SPECS[Aspect.RELEVANCE]
SUI = SPECS[Aspect.SUITABLENESS]


def test_parse_score_accepts_only_an_explicit_in_scale_integer():
    assert parse_score('{"score": 4}', REL) == 4
    assert parse_score('```json\n{"score": "5"}\n```', REL) == 5
    assert parse_score('Reasoning first. {"score": 2}', REL) == 2
    assert parse_score('{"score": 4}', SUI) is None
    assert parse_score('{"score": 3.5}', REL) is None
    assert parse_score('{"score": true}', REL) is None
    assert parse_score("I would give it a 4.", REL) is None
    assert parse_score('{"Relevance": 4}', REL) is None
    assert parse_score(None, REL) is None


def test_zero_shot_judge_scores_and_sends_no_steps():
    llm = FakeLLM(lambda request: '{"score": 3}')
    prediction = zero_shot_judge(llm).score(make_item("u1"), Aspect.COHERENCE)
    assert (prediction.score, prediction.attempts) == (3, 1)
    system = llm.calls[0].messages[0][1]
    assert "Argument Coherence (1-5)" in system and "Evaluation Steps" not in system


def test_judge_retries_once_then_reports_missing():
    replies = iter(["about a four", '{"score": 4}'])
    llm = FakeLLM(lambda request: next(replies))
    prediction = zero_shot_judge(llm).score(make_item("u1"), Aspect.RELEVANCE)
    assert (prediction.score, prediction.attempts) == (4, 2)
    assert llm.calls[1].messages[-1] == ("user", RETRY_MESSAGE)

    llm = FakeLLM(lambda request: "no idea")
    prediction = zero_shot_judge(llm).score(make_item("u2"), Aspect.RELEVANCE)
    assert (prediction.score, prediction.attempts) == (None, 2)


def test_auto_cseval_judge_uses_calibrated_steps():
    llm = FakeLLM(lambda request: '{"score": 1}')
    judge = auto_cseval_judge(llm, {Aspect.AGGRESSIVENESS: "1. Look for insults."})
    judge.score(make_item("u1"), Aspect.AGGRESSIVENESS)
    assert llm.calls[0].messages[0][1].endswith("## Evaluation Steps:\n1. Look for insults.")
