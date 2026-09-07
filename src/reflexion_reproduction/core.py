"""Core actor-evaluator-reflector-memory loop from the Reflexion paper."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Protocol, Sequence

from .metrics import exact_match, token_f1


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
    ) -> None:
        self.example = example
        self.model = model
        self.strategy = strategy
        self.memory: list[str] = []
        self.previous_trace = ""
        self.solved = False

    def run_trial(self, trial: int, can_reflect: bool) -> TrialRecord:
        memory_before = list(self.memory)
        output = self.model.act(
            example=self.example,
            memory=memory_before,
            previous_trace=self.previous_trace,
            attempt=trial,
        )
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
        )

        self.previous_trace = output.trace + f"\nObservation: Answer is {record.feedback}"
        self.solved = is_correct

        if not is_correct and can_reflect and self.strategy is Strategy.REFLEXION:
            reflection = self.model.reflect(
                example=self.example,
                failed_trace=self.previous_trace,
                memory=self.memory,
            ).strip()
            if reflection:
                self.memory.append(reflection)
                record.reflection_added = reflection
        return record


def run_experiment(
    examples: Sequence[Example],
    model: LanguageModel,
    strategy: Strategy,
    max_trials: int,
) -> ExperimentResult:
    """Evaluate cumulative task success across repeated trials."""

    if max_trials < 1:
        raise ValueError("max_trials must be at least 1")

    agents = [ReflexionAgent(example, model, strategy) for example in examples]
    result = ExperimentResult(
        strategy=strategy,
        provider=model.name,
        total_examples=len(examples),
        max_trials=max_trials,
    )

    for trial in range(1, max_trials + 1):
        for agent in agents:
            if agent.solved:
                continue
            result.records.append(
                agent.run_trial(trial=trial, can_reflect=trial < max_trials)
            )
    return result

