from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reflexion_reproduction.core import ActorOutput, Example  # noqa: E402
from reflexion_reproduction.data import (  # noqa: E402
    SealedSplitError,
    load_mini,
    load_split,
    row_to_example,
)
from reflexion_reproduction.experiment_cli import main as experiment_main  # noqa: E402
from reflexion_reproduction.failure_memory import (  # noqa: E402
    Lesson,
    MemoryBank,
    Retriever,
    build_memory,
    leaks_answer,
    parse_failure_label,
    parse_lesson,
    rule_failure_type,
)
from reflexion_reproduction.pipeline import analyze, mcnemar_exact, run_condition  # noqa: E402
from reflexion_reproduction.prompts import build_actor_prompt  # noqa: E402
from reflexion_reproduction.providers import ScriptedProvider  # noqa: E402


def make_example(example_id: str, qtype: str = "bridge", answer: str = "Paris") -> Example:
    return Example(
        id=example_id,
        question=f"Question {example_id} about a capital city?",
        answer=answer,
        context=({"title": "T", "text": "Some text."},),
        qtype=qtype,
        level="medium",
    )


def make_lesson(lesson_id: str, qtype: str, failure_type: str, source: str = "x") -> Lesson:
    return Lesson(
        id=lesson_id, source_example_id=source, source_question="capital city question",
        qtype=qtype, level="medium", failure_type=failure_type, failure_label_source="llm",
        diagnosis="d", fix=f"[{qtype}] check every clue", failed_prediction="p", validated=True,
    )


class DataTests(unittest.TestCase):
    def test_official_column_context_is_converted(self) -> None:
        example = row_to_example({
            "id": "1", "question": "q", "answer": "a", "type": "comparison", "level": "hard",
            "context": {"title": ["A", "B"], "sentences": [["x ", "y"], ["z"]]},
        })
        self.assertEqual(example.context[0], {"title": "A", "text": "x y"})
        self.assertEqual((example.qtype, example.level), ("comparison", "hard"))

    def test_mini_sample_still_loads(self) -> None:
        self.assertEqual(len(load_mini(ROOT / "data" / "hotpotqa_mini.json", 5)), 5)

    def test_validation_split_is_sealed(self) -> None:
        with self.assertRaises(SealedSplitError):
            load_split("validation", 3)

    def test_sampling_is_reproducible(self) -> None:
        first = [e.id for e in load_split("test", 5, seed=7)]
        second = [e.id for e in load_split("test", 5, seed=7)]
        other = [e.id for e in load_split("test", 5, seed=8)]
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)


class LabelingTests(unittest.TestCase):
    def test_rule_labels(self) -> None:
        self.assertEqual(rule_failure_type("", "Paris"), "NO_ANSWER")
        self.assertEqual(rule_failure_type("Cannot be determined", "Paris"), "NO_ANSWER")
        self.assertEqual(rule_failure_type("no", "yes"), "YES_NO_ERROR")
        self.assertEqual(rule_failure_type("the city of Paris France", "Paris"), "FORMAT_MISMATCH")
        self.assertIsNone(rule_failure_type("London", "Paris"))

    def test_label_parsing_falls_back_to_other(self) -> None:
        self.assertEqual(parse_failure_label("Label: intermediate_entity"), "INTERMEDIATE_ENTITY")
        self.assertEqual(parse_failure_label("no idea"), "OTHER")

    def test_lesson_parsing_and_leak_check(self) -> None:
        diagnosis, fix = parse_lesson("Diagnosis: stopped early.\nFix: follow the last hop.")
        self.assertEqual((diagnosis, fix), ("stopped early.", "follow the last hop."))
        self.assertTrue(leaks_answer("Answer with Paris next time", "Paris"))
        self.assertFalse(leaks_answer("Answer yes or no only", "yes"))


class PromptTests(unittest.TestCase):
    def test_no_lessons_keeps_original_prompt(self) -> None:
        example = make_example("a")
        self.assertNotIn("Lessons learned", build_actor_prompt(example, [], ""))
        self.assertIn("Lessons learned", build_actor_prompt(example, [], "", ["tip"]))


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bank = MemoryBank([
            make_lesson("L1", "bridge", "INTERMEDIATE_ENTITY"),
            make_lesson("L2", "bridge", "INTERMEDIATE_ENTITY"),
            make_lesson("L3", "bridge", "WRONG_ATTRIBUTE"),
            make_lesson("L4", "comparison", "COMPARISON_ERROR"),
            make_lesson("L5", "bridge", "FORMAT_MISMATCH", source="target"),
        ])
        self.bank.lessons.append(make_lesson("L6", "bridge", "NO_ANSWER"))
        self.bank.lessons[-1].validated = False

    def test_failure_type_retrieval_prefers_common_types_of_same_question_type(self) -> None:
        chosen = Retriever(self.bank, "failure_type", k=2).select(make_example("target"))
        types = [lesson.failure_type for lesson in chosen]
        self.assertEqual(types, ["INTERMEDIATE_ENTITY", "WRONG_ATTRIBUTE"])

    def test_retrieval_skips_own_task_and_unvalidated_lessons(self) -> None:
        for mode in ("failure_type", "random", "similarity"):
            ids = {l.id for l in Retriever(self.bank, mode, k=10).select(make_example("target"))}
            self.assertNotIn("L5", ids, mode)
            self.assertNotIn("L6", ids, mode)

    def test_retrieval_is_deterministic_per_seed(self) -> None:
        example = make_example("q1")
        a = Retriever(self.bank, "random", k=2, seed=1).select(example)
        b = Retriever(self.bank, "random", k=2, seed=1).select(example)
        self.assertEqual([l.id for l in a], [l.id for l in b])


class BuildMemoryTests(unittest.TestCase):
    def test_only_lessons_that_fix_the_task_are_validated(self) -> None:
        class StubbornModel(ScriptedProvider):
            def act(self, example, memory, previous_trace, attempt, lessons=()):
                return ActorOutput(thought="t", answer="wrong guess")

        examples = [make_example("never-easy-1"), make_example("never-easy-2")]
        bank, logs = build_memory(examples, StubbornModel())
        self.assertEqual(len(bank.lessons), 2)
        self.assertEqual(bank.validated(), [])
        self.assertTrue(all(l.rejected_reason == "fix did not change the outcome" for l in bank.lessons))
        self.assertEqual([log.trial for log in logs], ["1", "validation", "1", "validation"])

    def test_scripted_lessons_validate(self) -> None:
        bank, _ = build_memory([make_example("zz-hard", answer="Rome")], ScriptedProvider())
        self.assertTrue(all(lesson.validated for lesson in bank.lessons))


class AnalysisTests(unittest.TestCase):
    def test_mcnemar_exact(self) -> None:
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact(0, 8), 2 / 256)
        self.assertAlmostEqual(mcnemar_exact(3, 3), 1.0)

    def test_paired_flips_and_lesson_effects(self) -> None:
        examples = [make_example(f"e{i}", answer=f"ans{i}") for i in range(12)]
        bank = MemoryBank([make_lesson("L1", "bridge", "WRONG_ENTITY")])
        model = ScriptedProvider()
        rows = run_condition("no_memory", examples, model, 1, label="rules")
        rows += run_condition("failure_memory", examples, model, 1, bank=bank, label="rules")
        report = analyze(rows)
        comp = next(c for c in report["comparisons"]
                    if c["condition"] == "failure_memory" and c["metric"] == "first")
        failed_before = sum(1 for r in rows if r["condition"] == "no_memory" and not r["exact_match"])
        self.assertEqual(comp["fail_to_pass"], failed_before)
        self.assertEqual(comp["pass_to_fail"], 0)
        self.assertEqual(report["lesson_effects"]["failure_memory:L1"]["helped"], failed_before)


class CliTests(unittest.TestCase):
    def test_offline_pipeline_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = ["--out-dir", directory]
            self.assertEqual(experiment_main(["pilot", "--n", "6", *out]), 0)
            self.assertEqual(experiment_main(["build", "--n", "10", *out]), 0)
            self.assertEqual(experiment_main(["evaluate", "--n", "8", *out]), 0)
            self.assertEqual(
                experiment_main(["analyze", "--attempts", str(Path(directory) / "eval_attempts.csv")]), 0
            )
            self.assertEqual(experiment_main(["evaluate", "--split", "validation", "--n", "2", *out]), 1)
            self.assertEqual(experiment_main(["evaluate", "--split", "train", "--n", "2", *out]), 1)


if __name__ == "__main__":
    unittest.main()
