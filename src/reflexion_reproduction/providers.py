"""Deterministic and live language-model providers."""

from __future__ import annotations

import os
import re
import time
import zlib
from typing import Sequence

from .core import ActorOutput, Example
from .prompts import (
    CLASSIFY_INSTRUCTIONS,
    LESSON_INSTRUCTIONS,
    build_actor_prompt,
    build_reflection_prompt,
)


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
    last_usage: dict = {}

    def act(
        self,
        example: Example,
        memory: Sequence[str],
        previous_trace: str,
        attempt: int,
        lessons: Sequence[str] = (),
    ) -> ActorOutput:
        del previous_trace, attempt
        if memory:
            return ActorOutput(
                thought="I will apply the verbal memory and match the exact evidence.",
                answer=example.answer,
            )
        # Cross-task transfer fixture: a lesson written for the same question
        # type fixes the first attempt; lessons for other types do not.
        if example.qtype and any(f"[{example.qtype}]" in lesson for lesson in lessons):
            return ActorOutput(thought="A past lesson applies here.", answer=example.answer)
        # Outside the five mini items, roughly a third of questions are "easy".
        if example.id not in SCRIPTED_WRONG_ANSWERS and zlib.crc32(example.id.encode()) % 3 == 0:
            return ActorOutput(thought="Easy question.", answer=example.answer)
        return ActorOutput(
            thought="My first reading suggests this answer.",
            answer=SCRIPTED_WRONG_ANSWERS.get(example.id, "wrong guess"),
        )

    def complete(self, instructions: str, prompt: str) -> str:
        """Canned replies for the failure-classification and lesson prompts."""

        if instructions == CLASSIFY_INSTRUCTIONS:
            return "WRONG_ENTITY"
        if instructions == LESSON_INSTRUCTIONS:
            qtype = re.search(r"Question type: (\w+)", prompt)
            tag = qtype.group(1) if qtype else "general"
            return (
                "Diagnosis: The attempt chose an entity without checking every clue.\n"
                f"Fix: [{tag}] Confirm the answer satisfies every condition in the question."
            )
        return "OK"

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


class _LiveProvider:
    """Shared Actor/Reflection logic for live API providers."""

    name = "live"

    def __init__(self) -> None:
        self.last_usage: dict = {}
        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    def _generate(self, instructions: str, prompt: str) -> str:  # pragma: no cover
        raise NotImplementedError

    def _record_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        self.total_usage["prompt_tokens"] += prompt_tokens
        self.total_usage["completion_tokens"] += completion_tokens
        self.total_usage["calls"] += 1

    def act(
        self,
        example: Example,
        memory: Sequence[str],
        previous_trace: str,
        attempt: int,
        lessons: Sequence[str] = (),
    ) -> ActorOutput:
        del attempt
        raw = self._generate(
            instructions="You are the Actor component in a Reflexion experiment.",
            prompt=build_actor_prompt(example, memory, previous_trace, lessons),
        )
        try:
            return parse_actor_output(raw)
        except ValueError:
            # An unparseable reply counts as a failed attempt, not a crash.
            return ActorOutput(thought=raw.strip(), answer="")

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

    def complete(self, instructions: str, prompt: str) -> str:
        return self._generate(instructions=instructions, prompt=prompt)


def _with_retries(call, attempts: int = 6, base_delay: float = 2.0):
    """Retry rate-limit and transient server errors with exponential backoff."""

    for attempt in range(attempts):
        try:
            return call()
        except Exception as error:  # noqa: BLE001 - SDK raises several types
            status = getattr(error, "status_code", None)
            transient = status in (408, 409, 429, 500, 502, 503, 504) or type(
                error
            ).__name__ in ("APIConnectionError", "APITimeoutError")
            if not transient or attempt == attempts - 1:
                raise
            time.sleep(base_delay * (2**attempt))
    raise RuntimeError("unreachable")


class GroqProvider(_LiveProvider):
    """Live provider for open-weight models on GroqCloud's OpenAI-compatible API."""

    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        reasoning_effort: str | None = "low",
        max_tokens: int = 1024,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError(
                "GROQ_API_KEY is not set. Create a key at console.groq.com and "
                "export it (or add it to Colab Secrets) before using --provider groq."
            )
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "The Groq provider needs the openai package. "
                "Run: python -m pip install -r requirements.txt"
            ) from error

        self.model = model
        self.name = f"groq:{model}"
        self.temperature = temperature
        # Only GPT-OSS models accept reasoning settings on Groq.
        self.reasoning_effort = reasoning_effort if "gpt-oss" in model else None
        self.max_tokens = max_tokens
        self.seed = seed
        self.client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url=self.BASE_URL)

    def _generate(self, instructions: str, prompt: str) -> str:
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
        )
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.reasoning_effort:
            # Keep hidden reasoning out of the reply text; low effort leaves
            # room in the token budget for the actual answer.
            kwargs["reasoning_effort"] = self.reasoning_effort
            kwargs["extra_body"] = {"include_reasoning": False}
        try:
            response = _with_retries(lambda: self.client.chat.completions.create(**kwargs))
        except Exception as error:
            raise RuntimeError(f"Groq request failed: {error}") from error
        usage = getattr(response, "usage", None)
        self._record_usage(
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError(
                "The model returned no text. If this is a reasoning model, raise "
                "--max-tokens or lower --reasoning-effort."
            )
        return text


class OpenAIProvider(_LiveProvider):
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

        super().__init__()
        self.model = model
        self.name = f"openai:{model}"
        self.client = OpenAI()

    def _generate(self, instructions: str, prompt: str) -> str:
        try:
            response = _with_retries(
                lambda: self.client.responses.create(
                    model=self.model,
                    instructions=instructions,
                    input=prompt,
                    max_output_tokens=300,
                )
            )
        except Exception as error:
            raise RuntimeError(f"OpenAI request failed: {error}") from error
        usage = getattr(response, "usage", None)
        self._record_usage(
            int(getattr(usage, "input_tokens", 0) or 0),
            int(getattr(usage, "output_tokens", 0) or 0),
        )
        text = response.output_text.strip()
        if not text:
            raise RuntimeError("The model returned no text.")
        return text


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
