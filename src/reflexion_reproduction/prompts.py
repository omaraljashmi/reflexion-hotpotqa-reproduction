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
) -> str:
    memory_text = "\n".join(f"- {item}" for item in memory) or "(none)"
    previous = previous_trace or "(none — this is the first attempt)"
    return f"""{ACTOR_INSTRUCTIONS}

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

