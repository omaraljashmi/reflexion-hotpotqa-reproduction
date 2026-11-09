from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reflexion_reproduction.core import ActorOutput, Strategy, run_experiment  # noqa: E402
from reflexion_reproduction.data import load_mini  # noqa: E402
from reflexion_reproduction.pipeline import cumulative_accuracy  # noqa: E402
from reflexion_reproduction.providers import ScriptedProvider  # noqa: E402
from reflexion_reproduction.reproduction import (  # noqa: E402
    PAPER_REFERENCE,
    format_table,
    run_reproduction,
    save_reproduction,
)


class RecordingModel:
    """Fails every attempt and records what the actor prompt was given."""

    name = "recording"
    last_usage: dict = {}

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def act(self, example, memory, previous_trace, attempt, lessons=()):
        self.calls.append({"attempt": attempt, "memory": list(memory), "previous_trace": previous_trace})
        return ActorOutput(thought="guess", answer="wrong")

    def reflect(self, example, failed_trace, memory):
        return "Check the second hop before answering."


class BaselineIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.examples = load_mini(ROOT / "data" / "hotpotqa_mini.json", limit=1)

    def test_baseline_never_sees_earlier_attempts(self) -> None:
        model = RecordingModel()
        run_experiment(self.examples, model, Strategy.BASELINE, max_trials=3)
        self.assertEqual(len(model.calls), 3)
        self.assertTrue(all(call["previous_trace"] == "" for call in model.calls))
        self.assertTrue(all(call["memory"] == [] for call in model.calls))

    def test_reflexion_sees_failed_trace_and_reflection(self) -> None:
        model = RecordingModel()
        run_experiment(self.examples, model, Strategy.REFLEXION, max_trials=3)
        second = model.calls[1]
        self.assertIn("Answer is INCORRECT", second["previous_trace"])
        self.assertEqual(second["memory"], ["Check the second hop before answering."])


class CurveTests(unittest.TestCase):
    def test_cumulative_accuracy_counts_first_pass_only(self) -> None:
        rows = [
            {"phase": "eval", "condition": "reflexion", "run_id": 0, "example_id": "a", "trial": 1, "exact_match": 0},
            {"phase": "eval", "condition": "reflexion", "run_id": 0, "example_id": "a", "trial": 2, "exact_match": 1},
            {"phase": "eval", "condition": "reflexion", "run_id": 0, "example_id": "b", "trial": 1, "exact_match": 1},
            {"phase": "eval", "condition": "reflexion", "run_id": 0, "example_id": "c", "trial": 1, "exact_match": 0},
            {"phase": "eval", "condition": "reflexion", "run_id": 0, "example_id": "c", "trial": 2, "exact_match": 0},
            {"phase": "eval", "condition": "reflexion", "run_id": 0, "example_id": "c", "trial": 3, "exact_match": 0},
            {"phase": "eval", "condition": "no_memory", "run_id": 0, "example_id": "a", "trial": 1, "exact_match": 0},
        ]
        self.assertEqual(cumulative_accuracy(rows, "reflexion", 3), [1 / 3, 2 / 3, 2 / 3])
        self.assertEqual(cumulative_accuracy(rows, "no_memory", 3), [0.0, 0.0, 0.0])
        self.assertEqual(cumulative_accuracy(rows, "random_memory", 2), [0.0, 0.0])


class ReproductionRunTests(unittest.TestCase):
    def test_scripted_reproduction_end_to_end(self) -> None:
        examples = load_mini(ROOT / "data" / "hotpotqa_mini.json", limit=5)
        rows, summary = run_reproduction(examples, ScriptedProvider(), max_trials=3)
        self.assertEqual(summary["accuracy_by_attempt"]["no_memory"], [0.0, 0.0, 0.0])
        self.assertEqual(summary["accuracy_by_attempt"]["reflexion"], [0.0, 1.0, 1.0])
        self.assertEqual(summary["first_attempt_failures"], {"no_memory": 5, "reflexion": 5})
        self.assertEqual(summary["recovered_by_reflexion"], 5)
        self.assertIs(summary["paper_reference"], PAPER_REFERENCE)
        self.assertIn("Paper Reflexion", format_table(summary))
        with tempfile.TemporaryDirectory() as directory:
            paths = save_reproduction(Path(directory), rows, summary, {"n": 5})
            self.assertTrue(paths["attempts"].exists())
            self.assertTrue(paths["report"].exists())


if __name__ == "__main__":
    unittest.main()
