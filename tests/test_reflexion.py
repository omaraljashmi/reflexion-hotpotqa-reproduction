from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reflexion_reproduction.cli import load_examples, save_results  # noqa: E402
from reflexion_reproduction.core import Strategy, run_experiment  # noqa: E402
from reflexion_reproduction.metrics import exact_match, normalize_answer, token_f1  # noqa: E402
from reflexion_reproduction.providers import ScriptedProvider, parse_actor_output  # noqa: E402


class MetricsTests(unittest.TestCase):
    def test_hotpotqa_normalization(self) -> None:
        self.assertEqual(normalize_answer("The Creature-Comforts!"), "creaturecomforts")
        self.assertTrue(exact_match("A failed coup attempt.", "failed coup attempt"))
        self.assertAlmostEqual(token_f1("failed attempt", "a failed coup attempt"), 0.8)


class ProviderTests(unittest.TestCase):
    def test_actor_parser_accepts_paper_format(self) -> None:
        output = parse_actor_output("Thought: connect both facts\nFinish[2004]")
        self.assertEqual(output.thought, "connect both facts")
        self.assertEqual(output.answer, "2004")


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.examples = load_examples(ROOT / "data" / "hotpotqa_mini.json", limit=3)

    def test_reflexion_memory_improves_scripted_run(self) -> None:
        result = run_experiment(
            self.examples,
            ScriptedProvider(),
            Strategy.REFLEXION,
            max_trials=2,
        )
        self.assertEqual(result.accuracy_by_trial(), [0.0, 1.0])
        self.assertTrue(all(record.reflection_added for record in result.records[:3]))

    def test_baseline_has_no_verbal_memory(self) -> None:
        result = run_experiment(
            self.examples,
            ScriptedProvider(),
            Strategy.BASELINE,
            max_trials=2,
        )
        self.assertEqual(result.accuracy_by_trial(), [0.0, 0.0])
        self.assertTrue(all(not record.memory_before for record in result.records))

    def test_results_are_saved_as_json(self) -> None:
        result = run_experiment(
            self.examples[:1],
            ScriptedProvider(),
            Strategy.REFLEXION,
            max_trials=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            save_results(path, "scripted-offline", [result])
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["experiments"][0]["accuracy_by_trial"], [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()

