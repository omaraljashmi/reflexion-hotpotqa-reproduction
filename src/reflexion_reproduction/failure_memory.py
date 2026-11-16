"""Cross-task structured failure memory (the project's extension of Reflexion).

Reflexion stores a free-text reflection and reuses it only on the SAME task.
This module instead:

1. runs the Reflexion loop on each lesson question (attempt, right or wrong
   feedback, reflection, retry) and labels each first-attempt failure with a
   failure type (rules first, then the LLM);
2. reformulates that experience as a lesson with two separate parts: a
   task-specific *diagnosis* and a general *fix* that must not mention the
   task's entities or answer;
3. validates the lesson in two steps. First a fresh attempt at the question it
   came from must pass when it sees only the fix. That is not enough on its
   own, because the lesson may only suit that one question, so the lesson is
   then tried on separate development questions and kept only if it turns
   more wrong answers right than right answers wrong;
4. retrieves validated lessons for NEW tasks before their first attempt.

The dataset's answer is used only by the rule-based labeler and the scorer. It
is never placed in a prompt. When the agent recovers on a lesson question, the
lesson writer sees the answer the agent found by itself.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
import re
from typing import Callable, Protocol, Sequence

from .core import ActorOutput, Example, ReflexionAgent, Strategy, TrialRecord
from .metrics import exact_match, normalize_answer, token_f1
from .parallel import parallel_map
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

    def reflect(self, example: Example, failed_trace: str, memory: Sequence[str]) -> str: ...

    def complete(self, instructions: str, prompt: str) -> str: ...


# ---------------------------------------------------------------- labeling


def _letters_and_digits(text: str) -> str:
    return "".join(character for character in text if character.isalnum())


def _contains_tokens(longer: list[str], shorter: list[str]) -> bool:
    """True if ``shorter`` appears inside ``longer`` as consecutive words."""

    size = len(shorter)
    return bool(size) and any(
        longer[start : start + size] == shorter for start in range(len(longer) - size + 1)
    )


def rule_failure_type(prediction: str, reference: str) -> str | None:
    """Cheap, deterministic labels that need the reference answer (train/analysis only).

    FORMAT_MISMATCH means the right content in a different surface form: the
    same letters and digits with other punctuation ("Anglo‑Saxon" with a
    special hyphen), one answer inside the other ("Bradenton" for "Bradenton,
    Florida"), or mostly the same words. Sharing exactly half of the words is
    not enough: "Robert Mitchum" for "James Mitchum" is a different person.
    """

    pred = normalize_answer(prediction)
    ref = normalize_answer(reference)
    if not pred or NO_ANSWER_PATTERN.search(prediction):
        return "NO_ANSWER"
    if ref in ("yes", "no"):
        return "YES_NO_ERROR"
    if not ref:
        return None
    if _letters_and_digits(pred) == _letters_and_digits(ref):
        return "FORMAT_MISMATCH"
    pred_words, ref_words = pred.split(), ref.split()
    if _contains_tokens(pred_words, ref_words) or _contains_tokens(ref_words, pred_words):
        return "FORMAT_MISMATCH"
    if token_f1(prediction, reference) > 0.5:
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


REJECT_UNPARSEABLE = "unparseable lesson"
REJECT_LEAK = "fix quotes the answer"
REJECT_SPECIFIC = "fix names something from its question"
REJECT_NO_EFFECT = "fix did not change the outcome"
REJECT_NO_TRANSFER = "did not help other questions"

# Which lessons a retriever may draw from.
POOLS = ("validated", "retry", "unvalidated")


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
    # True once the lesson is in the shared pool. With development questions
    # that means it passed both checks below; without them, only the first.
    validated: bool = False
    validation_prediction: str = ""  # answer of the retry on its own question
    rejected_reason: str = ""
    # True if the agent's own Reflexion retries reached an accepted answer, so
    # the lesson describes a fix that worked at least once.
    recovered: bool = False
    # Check 1: a fresh attempt at its own question passes with only this lesson.
    fixes_source: bool = False
    # Check 2: tried on development questions that did not produce it.
    dev_questions: int = 0
    dev_helped: int = 0  # wrong without the lesson, right with it
    dev_hurt: int = 0  # right without the lesson, wrong with it

    @property
    def usable(self) -> bool:
        """It has a fix and the fix does not give away its question's answer."""

        return bool(self.fix) and self.rejected_reason not in (
            REJECT_UNPARSEABLE, REJECT_LEAK, REJECT_SPECIFIC
        )

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
    """Read the two labeled parts of a lesson, tolerating bold marks and line breaks."""

    cleaned = text.replace("*", "").replace("`", "")
    diagnosis = re.search(
        r"Diagnosis\s*:\s*(.+?)(?=\n\s*Fix\s*:|\Z)", cleaned, re.IGNORECASE | re.DOTALL
    )
    fix = re.search(r"Fix\s*:\s*(.+?)(?=\n\s*\n|\Z)", cleaned, re.IGNORECASE | re.DOTALL)
    return (
        " ".join(diagnosis.group(1).split()) if diagnosis else "",
        " ".join(fix.group(1).split()) if fix else "",
    )


def leaks_answer(fix: str, reference: str) -> bool:
    """True if the 'general' fix quotes the task's answer (yes/no answers exempt)."""

    ref = normalize_answer(reference)
    if not ref or ref in ("yes", "no"):
        return False
    return f" {ref} " in f" {normalize_answer(fix)} "


_NAME = re.compile(r"[A-Za-z][A-Za-z'\u2019-]+|\d{3,}")
# Capitalized words that are not names: question words and the lesson writer's own terms.
_NOT_A_NAME = frozenset(
    "What Which Who Whom Whose When Where Why How Are Is Was Were Did Do Does In On The A An Of "
    "And Both Between Has Have Had Can Could Would Will At For To From By With Yes No Name "
    "Month Day Year City State".split()
)


def _names(text: str) -> set[str]:
    """Capitalized words and numbers of three or more digits: the specifics of a question."""

    words = _NAME.findall(text)
    found = {
        word for index, word in enumerate(words)
        if word.isdigit()
        or (word[0].isupper() and len(word) > 2 and word not in _NOT_A_NAME
            # the first word of a sentence is capitalized by grammar alone
            and (index > 0 or word in words[1:]))
    }
    return found


def names_its_question(fix: str, question: str, first_answer: str) -> bool:
    """True if the 'general' fix repeats a name or number from its own question or first answer.

    Such a lesson only fits the question it came from ("write exactly the word
    Beatles"), so it is dropped before any check.
    """

    return bool((_names(question) | _names(first_answer)) & set(_NAME.findall(fix)))


@dataclass
class MemoryBank:
    lessons: list[Lesson] = field(default_factory=list)
    meta: dict = field(default_factory=dict)  # how the bank was built

    def validated(self) -> list[Lesson]:
        return [lesson for lesson in self.lessons if lesson.validated]

    def pool(self, name: str = "validated") -> list[Lesson]:
        """Lessons a retriever may use.

        validated: passed every check that was run (the shared pool).
        retry: passed the retry on their own question, with or without the
        development check. unvalidated: every usable lesson, checked or not.
        """

        if name == "validated":
            return self.validated()
        if name == "retry":
            return [lesson for lesson in self.lessons if lesson.fixes_source and lesson.usable]
        if name == "unvalidated":
            return [lesson for lesson in self.lessons if lesson.usable]
        raise ValueError(f"Unknown lesson pool {name!r}; choose from {POOLS}")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"meta": self.meta, "lessons": [asdict(lesson) for lesson in self.lessons]}
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "MemoryBank":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls([Lesson(**row) for row in payload["lessons"]], dict(payload.get("meta") or {}))


@dataclass
class BuildLog:
    """One row per model attempt during memory building (for the CSV dataset).

    ``trial`` says what the attempt was: "1", "2", ... are the Reflexion
    attempts at a lesson question, "validation" is the fresh retry of that
    question with only its lesson, "dev_baseline" a development question
    without lessons and "dev_check" a development question with one lesson.
    """

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
    reflection: str = ""


def _log_from_record(example: Example, record: TrialRecord) -> BuildLog:
    return BuildLog(
        example, str(record.trial), ActorOutput(record.thought, record.prediction),
        record.exact_match, record.f1,
        prompt_tokens=record.prompt_tokens, completion_tokens=record.completion_tokens,
        reflection=record.reflection_added or "",
    )


def _experience(records: Sequence[TrialRecord]) -> str:
    """The agent's attempts and reflections on one question, as shown to the lesson writer."""

    lines = []
    for record in records:
        lines.append(
            f"Attempt {record.trial}: Thought: {record.thought} Finish[{record.prediction}] "
            f"-> {record.feedback}"
        )
        if record.reflection_added:
            lines.append(f"Reflection after attempt {record.trial}: {record.reflection_added}")
    return "\n".join(lines)


def _usage(model: object) -> tuple[int, int]:
    usage = getattr(model, "last_usage", {}) or {}
    return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))


def _attempt(
    model: MemoryModel, example: Example, trial: str, lessons: Sequence[str] = ()
) -> BuildLog:
    """One fresh attempt: no reflections, no previous trace, only ``lessons``."""

    output = model.act(example, memory=(), previous_trace="", attempt=1, lessons=list(lessons))
    log = BuildLog(
        example, trial, output,
        exact_match(output.answer, example.answer), token_f1(output.answer, example.answer),
    )
    log.prompt_tokens, log.completion_tokens = _usage(model)
    return log


def build_memory(
    examples: Sequence[Example],
    model: MemoryModel,
    validate: bool = True,
    lesson_content: str = "fix",
    progress: Callable[[str], None] | None = None,
    dev_examples: Sequence[Example] = (),
    workers: int = 1,
    min_net: int = 1,
    max_trials: int = 3,
) -> tuple[MemoryBank, list[BuildLog]]:
    """Run Reflexion on each lesson question and turn the failures into checked lessons.

    ``examples`` are the questions lessons are written from; each gets up to
    ``max_trials`` attempts with reflection, exactly as in the parent paper.
    ``dev_examples`` are separate development questions used only to test
    lessons: a lesson that repaired its own question is then tried on each of
    them and enters the shared pool only if it helped at least ``min_net`` more
    answers than it hurt. Without development questions the retry on its own
    question decides. ``validate=False`` skips both checks.
    """

    def say(message: str) -> None:
        if progress:
            progress(message)

    def step(name: str) -> Callable[[int, int], None]:
        def report(done: int, total: int) -> None:
            if done == total or done % 25 == 0:
                say(f"[{name}] {done}/{total}")

        return report

    def learn(example: Example) -> tuple[list[BuildLog], Lesson | None]:
        # The parent paper's loop on this question: attempt, feedback, reflect, retry.
        agent = ReflexionAgent(example, model, Strategy.REFLEXION)
        records: list[TrialRecord] = []
        for trial in range(1, max_trials + 1):
            records.append(agent.run_trial(trial=trial, can_reflect=trial < max_trials))
            if records[-1].exact_match:
                break
        logs = [_log_from_record(example, record) for record in records]
        first = records[0]
        if first.exact_match:
            return logs, None

        failed_trace = (
            f"Thought: {first.thought}\nFinish[{first.prediction}]\nObservation: Answer is INCORRECT"
        )
        failure_type, source = classify_failure(model, example, first.prediction, failed_trace)
        logs[0].failure_type, logs[0].failure_label_source = failure_type, source
        recovered = records[-1].exact_match
        # The new step: reformulate the experience as a lesson for other questions.
        diagnosis, fix = parse_lesson(
            model.complete(
                LESSON_INSTRUCTIONS,
                build_lesson_prompt(example, _experience(records), failure_type, recovered),
            )
        )
        lesson = Lesson(
            id="",
            source_example_id=example.id,
            source_question=example.question,
            qtype=example.qtype,
            level=example.level,
            failure_type=failure_type,
            failure_label_source=source,
            diagnosis=diagnosis,
            fix=fix,
            failed_prediction=first.prediction,
            recovered=recovered,
        )
        if not fix:
            lesson.rejected_reason = REJECT_UNPARSEABLE
        elif leaks_answer(fix, example.answer):
            lesson.rejected_reason = REJECT_LEAK
        elif names_its_question(fix, example.question, first.prediction):
            lesson.rejected_reason = REJECT_SPECIFIC
        if not lesson.usable or not validate:
            return logs, lesson

        # Check 1: a FRESH attempt (no previous trace, no reflection) that sees
        # only this lesson. Passing means the lesson itself changed the outcome.
        retry = _attempt(model, example, "validation", [lesson.render(lesson_content)])
        lesson.fixes_source = retry.correct
        lesson.validation_prediction = retry.output.answer
        if not retry.correct:
            lesson.rejected_reason = REJECT_NO_EFFECT
        return logs + [retry], lesson

    bank = MemoryBank()
    logs: list[BuildLog] = []
    for example_logs, lesson in parallel_map(learn, examples, workers, on_done=step("lesson questions")):
        if lesson is not None:
            lesson.id = f"L{len(bank.lessons) + 1:04d}"
            bank.lessons.append(lesson)
            for log in example_logs:
                log.lesson_id = lesson.id
        logs.extend(example_logs)

    if not validate:
        for lesson in bank.lessons:
            lesson.validated = lesson.usable
        return bank, logs

    candidates = [lesson for lesson in bank.lessons if lesson.fixes_source]
    dev_examples = list(dev_examples)
    if not dev_examples:
        # No development questions: the retry on its own question decides.
        for lesson in candidates:
            lesson.validated = True
        return bank, logs

    # Check 2: does the lesson help on questions that did not produce it?
    say(f"Development check: {len(candidates)} lessons on {len(dev_examples)} separate questions")
    baseline = parallel_map(
        lambda example: _attempt(model, example, "dev_baseline"),
        dev_examples, workers, on_done=step("development baseline"),
    )
    logs.extend(baseline)
    pairs = [
        (lesson, index)
        for lesson in candidates
        for index, example in enumerate(dev_examples)
        if example.id != lesson.source_example_id
    ]

    def check(pair: tuple[Lesson, int]) -> BuildLog:
        lesson, index = pair
        log = _attempt(model, dev_examples[index], "dev_check", [lesson.render(lesson_content)])
        log.lesson_id = lesson.id
        return log

    checks = parallel_map(check, pairs, workers, on_done=step("development check"))
    logs.extend(checks)
    for (lesson, index), log in zip(pairs, checks):
        lesson.dev_questions += 1
        lesson.dev_helped += int(log.correct and not baseline[index].correct)
        lesson.dev_hurt += int(baseline[index].correct and not log.correct)
    for lesson in candidates:
        lesson.validated = lesson.dev_helped - lesson.dev_hurt >= min_net
        if not lesson.validated:
            lesson.rejected_reason = REJECT_NO_TRANSFER
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


def _similarity(question: str, lesson: Lesson) -> float:
    """Word overlap between a new question and the question a lesson came from."""

    query, other = _tokens(question), _tokens(lesson.source_question)
    union = query | other
    return len(query & other) / len(union) if union else 0.0


def _one_per_failure_type(lessons: Sequence[Lesson], question: str, limit: int) -> list[Lesson]:
    """One lesson per failure type, most common type first, repeated until ``limit``.

    Within a failure type the lesson from the most similar question goes first.
    """

    groups: dict[str, list[Lesson]] = defaultdict(list)
    for lesson in lessons:
        groups[lesson.failure_type].append(lesson)
    order = sorted(groups, key=lambda failure_type: (-len(groups[failure_type]), failure_type))
    for failure_type in order:
        groups[failure_type].sort(key=lambda lesson: (_similarity(question, lesson), lesson.id), reverse=True)
    chosen: list[Lesson] = []
    while len(chosen) < limit and any(groups[failure_type] for failure_type in order):
        for failure_type in order:
            if groups[failure_type] and len(chosen) < limit:
                chosen.append(groups[failure_type].pop(0))
    return chosen


class Retriever:
    """Selects up to k lessons for a new example before its first attempt.

    ``pool`` names which lessons may be used (see ``MemoryBank.pool``). Every
    mode returns the same number of lessons, min(k, pool size), so conditions
    differ in which lessons the agent reads, not in how many.
    """

    def __init__(
        self, bank: MemoryBank, mode: str, k: int = 3, seed: int = 0, pool: str = "validated"
    ) -> None:
        if mode not in ("failure_type", "random", "similarity"):
            raise ValueError(f"Unknown retrieval mode {mode!r}")
        self.pool = bank.pool(pool)
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
            ranked = sorted(pool, key=lambda lesson: (-_similarity(example.question, lesson), lesson.id))
            return ranked[: self.k]

        # failure_type: lessons from the same kind of question (bridge or
        # comparison), one per failure type starting with the type that kind of
        # question fails on most, so the agent reads about its most likely
        # mistakes. Other kinds of question only fill the remaining places.
        same_kind = [lesson for lesson in pool if lesson.qtype == example.qtype]
        chosen = _one_per_failure_type(same_kind, example.question, self.k)
        if len(chosen) < self.k:
            others = [lesson for lesson in pool if lesson.qtype != example.qtype]
            chosen += _one_per_failure_type(others, example.question, self.k - len(chosen))
        return chosen

    def lessons_for(self, content: str) -> Callable[[Example], tuple[list[str], list[str]]]:
        def _lookup(example: Example) -> tuple[list[str], list[str]]:
            selected = self.select(example)
            return [lesson.render(content) for lesson in selected], [lesson.id for lesson in selected]

        return _lookup
