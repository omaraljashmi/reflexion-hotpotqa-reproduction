"""Core actor-evaluator-reflector-memory loop from the Reflexion paper."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Callable, Protocol, Sequence

from .metrics import exact_match, token_f1
from .parallel import parallel_map


class Strategy(str, Enum):
    """Experimental condition used by the agent."""

    BASELINE = "baseline"
    REFLEXION = "reflexion"


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    answer: str
    context: tuple[dict[str, str], ...]
    qtype: str = ""
    level: str = ""

    @property
    def context_text(self) -> str:
        return "\n\n".join(
            f"[{item['title']}] {item['text']}" for item in self.context
        )


@dataclass(frozen=True)
class ActorOutput:
    thought: str
    answer: str

    @property
    def trace(self) -> str:
        return f"Thought: {self.thought}\nFinish[{self.answer}]"


class LanguageModel(Protocol):
    """The two language-model operations used by Reflexion."""

    name: str

    def act(
        self,
        example: Example,
        memory: Sequence[str],
        previous_trace: str,
        attempt: int,
        lessons: Sequence[str] = (),
    ) -> ActorOutput: ...

    def reflect(
        self,
        example: Example,
        failed_trace: str,
        memory: Sequence[str],
    ) -> str: ...


@dataclass
class TrialRecord:
    example_id: str
    trial: int
    thought: str
    prediction: str
    reference: str
    exact_match: bool
    f1: float
    feedback: str
    memory_before: list[str]
    reflection_added: str | None = None
    lessons_used: list[str] = field(default_factory=list)
    lesson_ids: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0  # response time of the Actor call
    reflection_tokens: int = 0  # tokens spent writing the reflection after this attempt


@dataclass
class ExperimentResult:
    strategy: Strategy
    provider: str
    total_examples: int
    max_trials: int
    records: list[TrialRecord] = field(default_factory=list)

    def solved_by_trial(self, trial: int) -> int:
        solved_ids = {
            record.example_id
            for record in self.records
            if record.trial <= trial and record.exact_match
        }
        return len(solved_ids)

    def accuracy_by_trial(self) -> list[float]:
        if self.total_examples == 0:
            return [0.0 for _ in range(self.max_trials)]
        return [
            self.solved_by_trial(trial) / self.total_examples
            for trial in range(1, self.max_trials + 1)
        ]

    @property
    def final_accuracy(self) -> float:
        return self.accuracy_by_trial()[-1] if self.max_trials else 0.0

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy.value,
            "provider": self.provider,
            "total_examples": self.total_examples,
            "max_trials": self.max_trials,
            "accuracy_by_trial": self.accuracy_by_trial(),
            "records": [asdict(record) for record in self.records],
        }


class ReflexionAgent:
    """Runs one task repeatedly and stores verbal reflections after failures."""

    def __init__(
        self,
        example: Example,
        model: LanguageModel,
        strategy: Strategy,
        lessons: Sequence[str] = (),
        lesson_ids: Sequence[str] = (),
        lessons_from_trial: int = 1,
    ) -> None:
        self.example = example
        self.model = model
        self.strategy = strategy
        # Cross-task lessons are fixed before the first attempt; within-task
        # reflections (self.memory) are the original Reflexion mechanism.
        self.lessons = list(lessons)
        self.lesson_ids = list(lesson_ids)
        # 1: the agent reads the lessons from its first attempt on. 2: only
        # once it has failed, so a first answer is never changed by a lesson.
        self.lessons_from_trial = lessons_from_trial
        self.memory: list[str] = []
        self.previous_trace = ""
        self.solved = False

    def run_trial(self, trial: int, can_reflect: bool) -> TrialRecord:
        memory_before = list(self.memory)
        # The baseline matches the paper's "CoT (GT) only" condition: every
        # attempt is a fresh prompt with no trace of earlier attempts. Only the
        # Reflexion agent carries its failed trial and reflections forward.
        previous = self.previous_trace if self.strategy is Strategy.REFLEXION else ""
        shown = trial >= self.lessons_from_trial
        lessons = self.lessons if shown else []
        output = self.model.act(
            example=self.example,
            memory=memory_before,
            previous_trace=previous,
            attempt=trial,
            lessons=lessons,
        )
        usage = dict(getattr(self.model, "last_usage", {}) or {})
        is_correct = exact_match(output.answer, self.example.answer)
        record = TrialRecord(
            example_id=self.example.id,
            trial=trial,
            thought=output.thought,
            prediction=output.answer,
            reference=self.example.answer,
            exact_match=is_correct,
            f1=token_f1(output.answer, self.example.answer),
            feedback="CORRECT" if is_correct else "INCORRECT",
            memory_before=memory_before,
            lessons_used=list(lessons),
            lesson_ids=list(self.lesson_ids) if shown else [],
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            seconds=float(usage.get("seconds", 0.0)),
        )

        self.previous_trace = output.trace + f"\nObservation: Answer is {record.feedback}"
        self.solved = is_correct

        if not is_correct and can_reflect and self.strategy is Strategy.REFLEXION:
            reflection = self.model.reflect(
                example=self.example,
                failed_trace=self.previous_trace,
                memory=self.memory,
            ).strip()
            used = dict(getattr(self.model, "last_usage", {}) or {})
            record.reflection_tokens = int(used.get("prompt_tokens", 0)) + int(
                used.get("completion_tokens", 0)
            )
            if reflection:
                self.memory.append(reflection)
                record.reflection_added = reflection
        return record


def run_experiment(
    examples: Sequence[Example],
    model: LanguageModel,
    strategy: Strategy,
    max_trials: int,
    lessons_for: Callable[[Example], tuple[Sequence[str], Sequence[str]]] | None = None,
    workers: int = 1,
    progress: Callable[[int, int, int], None] | None = None,
    lessons_from_trial: int = 1,
) -> ExperimentResult:
    """Evaluate cumulative task success across repeated trials.

    ``lessons_for`` optionally returns (lesson texts, lesson ids) retrieved from a
    cross-task memory bank before the first attempt on each example. ``workers``
    runs that many questions side by side; each question keeps its own agent and
    memory, so the results are the same as with one worker. ``progress(trial,
    done, total)`` is called after each finished attempt.
    """

    if max_trials < 1:
        raise ValueError("max_trials must be at least 1")

    agents = []
    for example in examples:
        texts, ids = lessons_for(example) if lessons_for else ((), ())
        agents.append(ReflexionAgent(example, model, strategy, texts, ids, lessons_from_trial))
    result = ExperimentResult(
        strategy=strategy,
        provider=model.name,
        total_examples=len(examples),
        max_trials=max_trials,
    )

    for trial in range(1, max_trials + 1):
        pending = [agent for agent in agents if not agent.solved]
        can_reflect = trial < max_trials
        result.records.extend(
            parallel_map(
                lambda agent: agent.run_trial(trial=trial, can_reflect=can_reflect),
                pending,
                workers,
                on_done=(lambda done, total: progress(trial, done, total)) if progress else None,
            )
        )
    return result

