"""Cross-task structured failure memory (the project's extension of Reflexion).

Reflexion stores a free-text reflection and reuses it only on the SAME task.
This module instead:

1. labels each failure with a failure type (rules first, then the LLM);
2. writes the lesson as two separate parts: a task-specific *diagnosis* and a
   general *fix* that must not mention the task's entities or answer;
3. keeps a lesson only if a fresh attempt that sees the fix now passes
   ("validated fix" — this shows the fix helped, not that the diagnosis is true);
4. retrieves lessons for NEW tasks before their first attempt.

The correct answer is used only by the rule-based labeler and the scorer. It is
never shown to the model in any prompt.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
import re
from typing import Callable, Protocol, Sequence

from .core import ActorOutput, Example
from .metrics import exact_match, normalize_answer, token_f1
from .prompts import (
    CLASSIFY_INSTRUCTIONS,
    FAILURE_TYPES,
    LESSON_INSTRUCTIONS,
    build_classify_prompt,
    build_lesson_prompt,
)


NO_ANSWER_PATTERN = re.compile(
    r"\b(unknown|not (?:stated|mentioned|given|provided|in the context)|cannot be determined|"
    r"insufficient|no information|unanswerable)\b",
    re.IGNORECASE,
)


class MemoryModel(Protocol):
    name: str

    def act(
        self,
        example: Example,
        memory: Sequence[str],
        previous_trace: str,
        attempt: int,
        lessons: Sequence[str] = (),
    ) -> ActorOutput: ...

    def complete(self, instructions: str, prompt: str) -> str: ...


# ---------------------------------------------------------------- labeling


def rule_failure_type(prediction: str, reference: str) -> str | None:
    """Cheap, deterministic labels that need the reference answer (train/analysis only)."""

    pred = normalize_answer(prediction)
    ref = normalize_answer(reference)
    if not pred or NO_ANSWER_PATTERN.search(prediction):
        return "NO_ANSWER"
    if ref in ("yes", "no"):
        return "YES_NO_ERROR"
    if token_f1(prediction, reference) >= 0.5 or (ref and (ref in pred or pred in ref)):
        return "FORMAT_MISMATCH"
    return None


def parse_failure_label(text: str) -> str:
    upper = text.upper()
    for label in FAILURE_TYPES:
        if label in upper:
            return label
    return "OTHER"


def classify_failure(
    model: MemoryModel,
    example: Example,
    prediction: str,
    failed_trace: str,
    use_llm: bool = True,
) -> tuple[str, str]:
    """Return (failure_type, source) where source is 'rule', 'llm', or 'none'."""

    label = rule_failure_type(prediction, example.answer)
    if label:
        return label, "rule"
    if not use_llm:
        return "OTHER", "none"
    reply = model.complete(CLASSIFY_INSTRUCTIONS, build_classify_prompt(example, failed_trace))
    return parse_failure_label(reply), "llm"


# ----------------------------------------------------------------- lessons


@dataclass
class Lesson:
    id: str
    source_example_id: str
    source_question: str
    qtype: str
    level: str
    failure_type: str
    failure_label_source: str
    diagnosis: str
    fix: str
    failed_prediction: str
    validated: bool = False
    validation_prediction: str = ""
    rejected_reason: str = ""

    def render(self, content: str = "fix") -> str:
        """Text shown to the Actor. 'fix' is the default because the diagnosis is
        task-specific; the 'diagnosis' and 'both' modes exist for the ablation."""

        meaning = FAILURE_TYPES.get(self.failure_type, FAILURE_TYPES["OTHER"])
        head = f"Common mistake ({self.failure_type}: {meaning})."
        if content == "fix":
            return f"{head} Fix: {self.fix}"
        if content == "diagnosis":
            return f"{head} Example of what went wrong: {self.diagnosis}"
        if content == "both":
            return f"{head} Example of what went wrong: {self.diagnosis} Fix: {self.fix}"
        raise ValueError(f"Unknown lesson content {content!r}")


def parse_lesson(text: str) -> tuple[str, str]:
    diagnosis = re.search(r"Diagnosis\s*:\s*(.+)", text, re.IGNORECASE)
    fix = re.search(r"Fix\s*:\s*(.+)", text, re.IGNORECASE)
    return (
        diagnosis.group(1).strip() if diagnosis else "",
        fix.group(1).strip() if fix else "",
    )


def leaks_answer(fix: str, reference: str) -> bool:
    """True if the 'general' fix quotes the task's answer (yes/no answers exempt)."""

    ref = normalize_answer(reference)
    if not ref or ref in ("yes", "no"):
        return False
    return f" {ref} " in f" {normalize_answer(fix)} "


@dataclass
class MemoryBank:
    lessons: list[Lesson] = field(default_factory=list)

    def validated(self) -> list[Lesson]:
        return [lesson for lesson in self.lessons if lesson.validated]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"lessons": [asdict(lesson) for lesson in self.lessons]}
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "MemoryBank":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls([Lesson(**row) for row in payload["lessons"]])


@dataclass
class BuildLog:
    """One row per model attempt during memory building (for the CSV dataset)."""

    example: Example
    trial: str
    output: ActorOutput
    correct: bool
    f1: float
    failure_type: str = ""
    failure_label_source: str = ""
    lesson_id: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _usage(model: object) -> tuple[int, int]:
    usage = getattr(model, "last_usage", {}) or {}
    return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))


def build_memory(
    examples: Sequence[Example],
    model: MemoryModel,
    validate: bool = True,
    lesson_content: str = "fix",
    progress: Callable[[str], None] | None = None,
) -> tuple[MemoryBank, list[BuildLog]]:
    """Attempt each training example once and turn failures into lessons."""

    bank = MemoryBank()
    logs: list[BuildLog] = []
    for index, example in enumerate(examples, start=1):
        output = model.act(example, memory=(), previous_trace="", attempt=1)
        correct = exact_match(output.answer, example.answer)
        log = BuildLog(example, "1", output, correct, token_f1(output.answer, example.answer))
        log.prompt_tokens, log.completion_tokens = _usage(model)
        logs.append(log)
        if progress:
            progress(f"[build {index}/{len(examples)}] {'PASS' if correct else 'FAIL'} {example.id}")
        if correct:
            continue

        failed_trace = f"{output.trace}\nObservation: Answer is INCORRECT"
        failure_type, source = classify_failure(model, example, output.answer, failed_trace)
        log.failure_type, log.failure_label_source = failure_type, source

        diagnosis, fix = parse_lesson(
            model.complete(LESSON_INSTRUCTIONS, build_lesson_prompt(example, failed_trace, failure_type))
        )
        lesson = Lesson(
            id=f"L{len(bank.lessons) + 1:04d}",
            source_example_id=example.id,
            source_question=example.question,
            qtype=example.qtype,
            level=example.level,
            failure_type=failure_type,
            failure_label_source=source,
            diagnosis=diagnosis,
            fix=fix,
            failed_prediction=output.answer,
        )
        log.lesson_id = lesson.id
        bank.lessons.append(lesson)

        if not fix:
            lesson.rejected_reason = "unparseable lesson"
            continue
        if leaks_answer(fix, example.answer):
            lesson.rejected_reason = "fix quotes the answer"
            continue
        if not validate:
            lesson.validated = True
            continue

        # Validation: a FRESH attempt (no previous trace, no reflection) that sees
        # only this lesson. Passing means the lesson itself changed the outcome.
        retry = model.act(
            example,
            memory=(),
            previous_trace="",
            attempt=1,
            lessons=[lesson.render(lesson_content)],
        )
        retry_correct = exact_match(retry.answer, example.answer)
        lesson.validated = retry_correct
        lesson.validation_prediction = retry.answer
        if not retry_correct:
            lesson.rejected_reason = "fix did not change the outcome"
        vlog = BuildLog(
            example, "validation", retry, retry_correct, token_f1(retry.answer, example.answer),
            lesson_id=lesson.id,
        )
        vlog.prompt_tokens, vlog.completion_tokens = _usage(model)
        logs.append(vlog)
    return bank, logs


# --------------------------------------------------------------- retrieval

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "of", "in", "on", "and", "or", "is", "was", "what", "which",
    "who", "whom", "when", "where", "how", "did", "does", "do", "to", "for", "by",
    "that", "this", "with", "as", "at", "from", "are", "were", "be", "it", "its",
}


def _tokens(text: str) -> set[str]:
    return {word for word in _WORD.findall(text.lower()) if word not in _STOP}


class Retriever:
    """Selects up to k validated lessons for a new example before its first attempt."""

    def __init__(self, bank: MemoryBank, mode: str, k: int = 3, seed: int = 0) -> None:
        if mode not in ("failure_type", "random", "similarity"):
            raise ValueError(f"Unknown retrieval mode {mode!r}")
        self.pool = bank.validated()
        self.mode = mode
        self.k = k
        self.seed = seed

    def _rng(self, example: Example) -> random.Random:
        return random.Random(f"{self.seed}:{self.mode}:{example.id}")

    def select(self, example: Example) -> list[Lesson]:
        pool = [lesson for lesson in self.pool if lesson.source_example_id != example.id]
        if not pool or self.k < 1:
            return []
        rng = self._rng(example)

        if self.mode == "random":
            return rng.sample(pool, min(self.k, len(pool)))

        if self.mode == "similarity":
            query = _tokens(example.question)

            def score(lesson: Lesson) -> float:
                other = _tokens(lesson.source_question)
                union = query | other
                return len(query & other) / len(union) if union else 0.0

            ranked = sorted(pool, key=lambda lesson: (-score(lesson), lesson.id))
            return ranked[: self.k]

        # failure_type: the failure types most common for this question type,
        # one lesson per type, so the agent sees its most likely mistakes.
        same_type = [lesson for lesson in pool if lesson.qtype == example.qtype] or pool
        counts = Counter(lesson.failure_type for lesson in same_type)
        chosen: list[Lesson] = []
        for failure_type, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            candidates = [lesson for lesson in same_type if lesson.failure_type == failure_type]
            chosen.append(rng.choice(candidates))
            if len(chosen) == self.k:
                break
        return chosen

    def lessons_for(self, content: str) -> Callable[[Example], tuple[list[str], list[str]]]:
        def _lookup(example: Example) -> tuple[list[str], list[str]]:
            selected = self.select(example)
            return [lesson.render(content) for lesson in selected], [lesson.id for lesson in selected]

        return _lookup
