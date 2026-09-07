"""Deterministic and live language-model providers."""

from __future__ import annotations

import os
import re
from typing import Sequence

from .core import ActorOutput, Example
from .prompts import build_actor_prompt, build_reflection_prompt


SCRIPTED_WRONG_ANSWERS = {
    "5a7180205542994082a3e856": "Shaun the Sheep",
    "5a8e27d45542995a26add46a": "2009",
    "5a881d2355429938390d3eeb": "Rascal Flatts",
    "5abdd0f15542991f6610604d": "an assassination",
    "5a78bc6b554299148911f979": "monthly lifestyle magazines",
}

SCRIPTED_REFLECTIONS = {
    "5a7180205542994082a3e856": (
        "I answered with another Nick Park creation instead of matching the clue about "
        "animated zoo animals. Next time I will follow that exact description in the context."
    ),
    "5a8e27d45542995a26add46a": (
        "I identified the band but did not retrieve its stated disbanding year. Next time I "
        "will connect Scott Stapp to Creed and copy the year given in Creed's paragraph."
    ),
    "5a881d2355429938390d3eeb": (
        "I guessed a country act without checking both clues. Next time I will find the act "
        "that both covered the song and was affiliated with Lyric Street Records."
    ),
    "5abdd0f15542991f6610604d": (
        "I named the event too generally. Next time I will use the full description of the "
        "circumstances surrounding Dollfuss's assassination."
    ),
    "5a78bc6b554299148911f979": (
        "I ignored the shared publication frequency and audience. Next time I will combine "
        "the attributes common to both magazine descriptions."
    ),
}


class ScriptedProvider:
    """Offline fixture that deterministically exercises the complete control flow.

    This provider is a smoke test, not a scientific model evaluation. It deliberately
    fails the first attempt and follows its stored reflection on the next attempt.
    """

    name = "scripted-offline"

    def act(
        self,
        example: Example,
        memory: Sequence[str],
        previous_trace: str,
        attempt: int,
    ) -> ActorOutput:
        del previous_trace, attempt
        if memory:
            return ActorOutput(
                thought="I will apply the verbal memory and match the exact evidence.",
                answer=example.answer,
            )
        return ActorOutput(
            thought="My first reading suggests this answer.",
            answer=SCRIPTED_WRONG_ANSWERS.get(example.id, "unknown"),
        )

    def reflect(
        self,
        example: Example,
        failed_trace: str,
        memory: Sequence[str],
    ) -> str:
        del failed_trace, memory
        return SCRIPTED_REFLECTIONS.get(
            example.id,
            "I should re-check the context and return the shortest exact answer next time.",
        )


class OpenAIProvider:
    """Live provider using the current OpenAI Responses API."""

    def __init__(self, model: str) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your terminal environment "
                "before using --provider openai."
            )
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "The OpenAI provider needs the optional dependency. "
                "Run: python -m pip install -r requirements.txt"
            ) from error

        self.model = model
        self.name = f"openai:{model}"
        self.client = OpenAI()

    def _generate(self, instructions: str, prompt: str) -> str:
        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=instructions,
                input=prompt,
                max_output_tokens=300,
            )
        except Exception as error:
            raise RuntimeError(f"OpenAI request failed: {error}") from error
        text = response.output_text.strip()
        if not text:
            raise RuntimeError("The model returned no text.")
        return text

    def act(
        self,
        example: Example,
        memory: Sequence[str],
        previous_trace: str,
        attempt: int,
    ) -> ActorOutput:
        del attempt
        raw = self._generate(
            instructions="You are the Actor component in a Reflexion experiment.",
            prompt=build_actor_prompt(example, memory, previous_trace),
        )
        return parse_actor_output(raw)

    def reflect(
        self,
        example: Example,
        failed_trace: str,
        memory: Sequence[str],
    ) -> str:
        return self._generate(
            instructions="You are the Self-Reflection component in a Reflexion experiment.",
            prompt=build_reflection_prompt(example, failed_trace, memory),
        )


def parse_actor_output(text: str) -> ActorOutput:
    """Parse a model reply while tolerating minor formatting differences."""

    finish_matches = re.findall(r"Finish\s*\[([^\]]+)\]", text, flags=re.IGNORECASE)
    if finish_matches:
        answer = finish_matches[-1].strip()
    else:
        answer_match = re.search(r"(?:Final\s+)?Answer\s*:\s*(.+)", text, re.IGNORECASE)
        if not answer_match:
            raise ValueError(f"Could not parse a final answer from model output: {text!r}")
        answer = answer_match.group(1).strip()

    thought_match = re.search(
        r"Thought\s*:\s*(.*?)(?=\n\s*(?:Finish|(?:Final\s+)?Answer)\b)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    thought = thought_match.group(1).strip() if thought_match else text.strip()
    return ActorOutput(thought=thought, answer=answer)
