"""Prompts adapted from the authors' HotPotQA chain-of-thought experiment."""

from __future__ import annotations

from typing import Sequence

from .core import Example


ACTOR_INSTRUCTIONS = """Solve the question using only the supplied context.
First state one concise Thought that identifies the relevant evidence and relation.
Then give one short final answer in exactly this form: Finish[answer]
Do not add text after Finish[answer]."""

REFLECTION_INSTRUCTIONS = """You are a reasoning agent that improves from verbal feedback.
The previous answer failed exact-match evaluation. In two concise sentences, diagnose the
likely reasoning or phrasing error and give a specific plan for the next attempt. Do not
state that you saw the answer key; it was not provided."""


def build_actor_prompt(
    example: Example,
    memory: Sequence[str],
    previous_trace: str,
    lessons: Sequence[str] = (),
) -> str:
    memory_text = "\n".join(f"- {item}" for item in memory) or "(none)"
    previous = previous_trace or "(none — this is the first attempt)"
    # The lessons block is added only when lessons exist, so the no-memory and
    # Reflexion conditions keep the original reproduction prompt unchanged.
    lessons_block = ""
    if lessons:
        lesson_lines = "\n".join(f"- {item}" for item in lessons)
        lessons_block = f"""
Lessons from earlier mistakes on other questions (follow a lesson only when it fits this question):
{lesson_lines}
"""
    return f"""{ACTOR_INSTRUCTIONS}
{lessons_block}
Relevant context:
{example.context_text}

Question: {example.question}

Verbal memory from earlier failed trials:
{memory_text}

Previous trial:
{previous}
"""


def build_reflection_prompt(
    example: Example,
    failed_trace: str,
    memory: Sequence[str],
) -> str:
    earlier_memory = "\n".join(f"- {item}" for item in memory) or "(none)"
    return f"""{REFLECTION_INSTRUCTIONS}

Relevant context:
{example.context_text}

Question: {example.question}

Earlier verbal memory:
{earlier_memory}

Failed trial and evaluator feedback:
{failed_trace}

Reflection:
"""



FAILURE_TYPES = {
    "FORMAT_MISMATCH": "the answer had the right content but extra, missing, or differently phrased words",
    "YES_NO_ERROR": "a yes/no question was answered with the wrong polarity or not with yes/no",
    "NO_ANSWER": "the agent said the answer was unknown or not in the context",
    "INTERMEDIATE_ENTITY": "a multi-hop question was answered with the bridge entity instead of following the final hop",
    "WRONG_ATTRIBUTE": "the right entity was found but the wrong property of it was reported",
    "COMPARISON_ERROR": "two items were compared incorrectly or the wrong side was chosen",
    "NUMERIC_DATE_ERROR": "a number, date, count, or calculation was wrong",
    "WRONG_ENTITY": "the answer named an unrelated or distractor entity",
    "OTHER": "none of the above",
}

CLASSIFY_INSTRUCTIONS = """You label why a question-answering attempt failed.
Choose exactly one label from the list. Reply with only the label."""

LESSON_INSTRUCTIONS = """You turn one experience of a question-answering agent into a lesson for OTHER questions.
The agent's answers were graded by exact match against a short reference answer. After each answer it was told only whether it was right or wrong.

Write exactly two lines:
Diagnosis: one sentence on what was wrong with the first answer in THIS experience. If a later answer was accepted, say how the accepted answer differed from the first one.
Fix: one rule for OTHER questions of the same kind. Start with "When the question asks for" and name the kind of thing asked for (such as a date, a person, a place, a number, or a yes or no decision), then say exactly what to write or what to check.

Rules for the Fix:
- It must be concrete enough to change an answer. Do not write "match the expected format" or "be precise".
- It must not mention any specific person, place, title, date, or answer from this question, and it must not give examples."""


def build_classify_prompt(example: Example, failed_trace: str) -> str:
    labels = "\n".join(f"{name}: {meaning}" for name, meaning in FAILURE_TYPES.items())
    return f"""Labels:
{labels}

Question type: {example.qtype or "unknown"}
Question: {example.question}

Relevant context:
{example.context_text}

Failed attempt:
{failed_trace}

Label:"""


def build_lesson_prompt(example: Example, experience: str, failure_type: str, recovered: bool) -> str:
    """``experience`` is the agent's own attempts and reflections on this question.

    The dataset's answer is never included. When the agent recovered, the
    accepted answer in the experience is one the agent produced itself.
    """

    meaning = FAILURE_TYPES.get(failure_type, FAILURE_TYPES["OTHER"])
    outcome = (
        "The last answer above was accepted." if recovered else "No answer was accepted."
    )
    return f"""{LESSON_INSTRUCTIONS}

Failure type of the first answer: {failure_type} ({meaning})
Question type: {example.qtype or "unknown"}
Question: {example.question}

Relevant context:
{example.context_text}

Experience:
{experience}
{outcome}
"""
