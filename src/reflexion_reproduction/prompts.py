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
Lessons learned from mistakes on other, unrelated questions (apply only if relevant):
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

LESSON_INSTRUCTIONS = """You write reusable lessons for a question-answering agent.
The agent's answer failed exact-match evaluation. You are not told the correct answer.
Write exactly two lines:
Diagnosis: one sentence on what likely went wrong in THIS attempt.
Fix: one general rule the agent can apply to OTHER questions to avoid this kind of mistake.
The Fix must not mention any specific person, place, title, date, or answer from this question."""


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


def build_lesson_prompt(example: Example, failed_trace: str, failure_type: str) -> str:
    meaning = FAILURE_TYPES.get(failure_type, FAILURE_TYPES["OTHER"])
    return f"""{LESSON_INSTRUCTIONS}

Failure type: {failure_type} ({meaning})
Question type: {example.qtype or "unknown"}
Question: {example.question}

Relevant context:
{example.context_text}

Failed attempt:
{failed_trace}
"""
