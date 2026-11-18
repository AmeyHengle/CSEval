from cseval.aspects import Aspect
from cseval.prompts import drafting_messages, geval_steps_messages, refinement_messages, scoring_messages

from .conftest import make_item

FIG5_INSTRUCTION = (
    "You will be given one counterspeech (also called as counter-argument or counter-narrative) written for a "
    "hateful statement (hate speech). Your task is to rate the counterspeech on one metric. Please make sure you "
    "read and understand these instructions carefully."
)


def test_drafting_prompt_is_figure_4():
    item = make_item("u1", aggressiveness=5.0)
    ((role, text),) = drafting_messages(Aspect.AGGRESSIVENESS, [item])
    assert role == "user"
    assert text.startswith("## Objective:")
    assert "and an Aggressiveness score (1-5) rated by a human expert" in text
    assert "## Evaluation Criteria for Aggressiveness:\nAggressiveness (1-5) - This metric assesses" in text
    assert "influence the Aggressiveness rating. Then, create a detailed set of steps" in text
    assert text.endswith("Expert Rating (Aggressiveness): 5.0")
    assert "$" not in text


def test_drafting_prompt_uses_the_requested_aspect():
    item = make_item("u1", relevance=2.0, coherence=4.0)
    ((_, text),) = drafting_messages(Aspect.COHERENCE, [item])
    assert "a Coherence score (1-5)" in text
    assert "Expert Rating (Coherence): 4.0" in text
    assert "Relevance" not in text


def test_scoring_prompt_is_figure_5():
    item = make_item("u1")
    (sys_role, system), (user_role, user) = scoring_messages(Aspect.SUITABLENESS, item, "1. Read.\n2. Rate.")
    assert (sys_role, user_role) == ("system", "user")
    assert FIG5_INSTRUCTION in system
    assert "## Evaluation Criteria\nSuitableness (1-3) - This metric measures" in system
    assert system.endswith("## Evaluation Steps:\n1. Read.\n2. Rate.")
    assert user.startswith("Hate Speech: Women are too emotional to be leaders.\nCounterspeech: Counterspeech u1.")
    assert user.endswith('{"score": <integer from 1 to 3>}.')


def test_zero_shot_prompt_has_no_steps():
    (_, system), _ = scoring_messages(Aspect.RELEVANCE, make_item("u1"), None)
    assert "Evaluation Steps" not in system


def test_refinement_prompt_is_figure_6():
    item = make_item("u1", relevance=5.0)
    ((_, text),) = refinement_messages(Aspect.RELEVANCE, "1. Old step.", [(item, 2)])
    assert text.startswith("Please refine and improve the chain-of-thought (CoT) evaluation steps")
    assert "in evaluating Relevance of counterspeech generation." in text
    assert "Please return only your refined criteria without any additional sentences." in text
    assert "Old criteria: 1. Old step." in text
    assert "Expert Rating (Relevance): 5.0\nLLM Rating (Relevance): 2" in text


def test_geval_prompt_contains_the_criteria():
    ((_, text),) = geval_steps_messages(Aspect.COHERENCE)
    assert "Argument Coherence (1-5)" in text and FIG5_INSTRUCTION in text
