from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import threading
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reflexion_reproduction.core import ActorOutput, Example, Strategy, run_experiment  # noqa: E402
from reflexion_reproduction.data import (  # noqa: E402
    SealedSplitError,
    load_mini,
    load_split,
    row_to_example,
)
from reflexion_reproduction.experiment_cli import main as experiment_main  # noqa: E402
from reflexion_reproduction.failure_memory import (  # noqa: E402
    REJECT_NO_EFFECT,
    REJECT_NO_TRANSFER,
    REJECT_SPECIFIC,
    Lesson,
    MemoryBank,
    Retriever,
    build_memory,
    leaks_answer,
    names_its_question,
    parse_failure_label,
    parse_lesson,
    rule_failure_type,
)
from reflexion_reproduction.figures import plot_all  # noqa: E402
from reflexion_reproduction.metrics import exact_match  # noqa: E402
from reflexion_reproduction.parallel import TokenBudget, parallel_map  # noqa: E402
from reflexion_reproduction.pipeline import (  # noqa: E402
    CONDITIONS,
    analyze,
    comparison_table,
    mcnemar_exact,
    paired_f1,
    paired_gain,
    read_rows,
    run_condition,
    wilson_interval,
)
from reflexion_reproduction.prompts import (  # noqa: E402
    CLASSIFY_INSTRUCTIONS,
    build_actor_prompt,
    build_lesson_prompt,
)
from reflexion_reproduction.providers import ScriptedProvider, _LiveProvider, _with_retries  # noqa: E402


def make_example(example_id: str, qtype: str = "bridge", answer: str = "Paris") -> Example:
    return Example(
        id=example_id,
        question=f"Question {example_id} about a capital city?",
        answer=answer,
        context=({"title": "T", "text": "Some text."},),
        qtype=qtype,
        level="medium",
    )


def make_lesson(
    lesson_id: str, qtype: str, failure_type: str, source: str = "x",
    question: str = "capital city question",
) -> Lesson:
    return Lesson(
        id=lesson_id, source_example_id=source, source_question=question,
        qtype=qtype, level="medium", failure_type=failure_type, failure_label_source="llm",
        diagnosis="d", fix=f"[{qtype}] check every clue", failed_prediction="p",
        validated=True, fixes_source=True,
    )


class TaggedModel:
    """A controllable stand-in for the model.

    Questions whose id starts with "ok" are right on the first attempt. Any
    other question is wrong until the agent has a reflection in memory or
    reads a lesson containing GOOD. A lesson containing BAD breaks the "ok"
    questions whose id contains "fragile".
    """

    name = "tagged"
    last_usage: dict = {}

    def act(self, example, memory, previous_trace, attempt, lessons=()):
        text = " ".join(lessons)
        if "BAD" in text and "fragile" in example.id:
            return ActorOutput("t", "wrong")
        if example.id.startswith("ok") or memory or "GOOD" in text:
            return ActorOutput("t", example.answer)
        return ActorOutput("t", "wrong")

    def reflect(self, example, failed_trace, memory):
        return "Look again."

    def complete(self, instructions, prompt):
        if instructions == CLASSIFY_INSTRUCTIONS:
            return "WRONG_ENTITY"
        rule = "GOOD BAD rule" if "harmful" in prompt else "GOOD rule"
        return f"**Diagnosis:** the first answer was wrong.\n**Fix:** {rule}"


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

    def test_gold_context_keeps_only_supporting_paragraphs(self) -> None:
        full = load_split("test", 5, seed=3)
        gold = load_split("test", 5, seed=3, gold_only=True)
        self.assertEqual([e.id for e in full], [e.id for e in gold])
        for f, g in zip(full, gold):
            self.assertLess(len(g.context), len(f.context))
            self.assertTrue(all(item in f.context for item in g.context))
            self.assertEqual(g.answer, f.answer)

    def test_level_filter(self) -> None:
        hard = load_split("test", 8, seed=1, level="hard")
        self.assertEqual(len(hard), 8)
        self.assertTrue(all(e.level == "hard" for e in hard))
        with self.assertRaises(ValueError):
            load_split("test", 2, level="impossible")

    def test_sampling_is_reproducible(self) -> None:
        first = [e.id for e in load_split("test", 5, seed=7)]
        second = [e.id for e in load_split("test", 5, seed=7)]
        other = [e.id for e in load_split("test", 5, seed=8)]
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)


class GradingTests(unittest.TestCase):
    def test_typographic_punctuation_is_not_an_error(self) -> None:
        # Non-breaking hyphen, curly quotes and narrow spaces, as the model writes them.
        self.assertTrue(exact_match("Kim Ha‑neul", "Kim Ha-neul"))
        self.assertTrue(exact_match("Alfrederick “Al” Joyner", 'Alfrederick "Al" Joyner'))
        self.assertFalse(exact_match("Bradenton", "Bradenton, Florida"))


class LabelingTests(unittest.TestCase):
    def test_rule_labels(self) -> None:
        self.assertEqual(rule_failure_type("", "Paris"), "NO_ANSWER")
        self.assertEqual(rule_failure_type("Cannot be determined", "Paris"), "NO_ANSWER")
        self.assertEqual(rule_failure_type("no", "yes"), "YES_NO_ERROR")
        self.assertEqual(rule_failure_type("the city of Paris France", "Paris"), "FORMAT_MISMATCH")
        self.assertIsNone(rule_failure_type("London", "Paris"))

    def test_format_mismatch_means_same_content(self) -> None:
        # One answer inside the other, or mostly the same words.
        self.assertEqual(rule_failure_type("Bradenton", "Bradenton, Florida"), "FORMAT_MISMATCH")
        self.assertEqual(
            rule_failure_type("Wentworth Miller", "Wentworth Earl Miller III"), "FORMAT_MISMATCH"
        )
        # Half the words in common is a different answer, not a different format.
        self.assertIsNone(rule_failure_type("Robert Mitchum", "James Mitchum"))
        self.assertIsNone(
            rule_failure_type("Polish-Russian War of 1792", "the Russo-Turkish War of 1787-92")
        )
        # A substring of a word is not the word.
        self.assertIsNone(rule_failure_type("art", "Stuart"))

    def test_label_parsing_falls_back_to_other(self) -> None:
        self.assertEqual(parse_failure_label("Label: intermediate_entity"), "INTERMEDIATE_ENTITY")
        self.assertEqual(parse_failure_label("no idea"), "OTHER")

    def test_lesson_parsing_and_leak_check(self) -> None:
        diagnosis, fix = parse_lesson("Diagnosis: stopped early.\nFix: follow the last hop.")
        self.assertEqual((diagnosis, fix), ("stopped early.", "follow the last hop."))
        self.assertTrue(leaks_answer("Answer with Paris next time", "Paris"))
        self.assertFalse(leaks_answer("Answer yes or no only", "yes"))

    def test_a_fix_may_not_name_its_own_question(self) -> None:
        question = "What rock band was named after the group that Buddy Holly fronted?"
        self.assertTrue(names_its_question("Write exactly the word Beatles.", question, "The Beatles"))
        self.assertTrue(names_its_question("Check what Holly fronted.", question, "wrong"))
        self.assertTrue(names_its_question("Include the 1994 season.", "Who missed games in 1994?", "18"))
        general = "When the question asks for a band, write its name as the passage does."
        self.assertFalse(names_its_question(general, question, "The Beatles"))
        # A capital letter at the start of the question is grammar, not a name.
        self.assertFalse(names_its_question("During retries, check again.", "During which war was it built?", "x"))

    def test_lesson_parsing_tolerates_bold_and_line_breaks(self) -> None:
        text = "**Diagnosis:** gave only\nthe year.\n**Fix:** When the question asks for a date,\nwrite it in full.\n\nExtra note."
        self.assertEqual(
            parse_lesson(text),
            ("gave only the year.", "When the question asks for a date, write it in full."),
        )
        self.assertEqual(parse_lesson("no labels here"), ("", ""))


class PromptTests(unittest.TestCase):
    def test_no_lessons_keeps_original_prompt(self) -> None:
        example = make_example("a")
        self.assertNotIn("Lessons from earlier mistakes", build_actor_prompt(example, [], ""))
        self.assertIn("Lessons from earlier mistakes", build_actor_prompt(example, [], "", ["tip"]))

    def test_lesson_prompt_never_contains_the_dataset_answer(self) -> None:
        example = make_example("a", answer="Canberra")
        prompt = build_lesson_prompt(example, "Attempt 1: Finish[Sydney] -> INCORRECT", "WRONG_ENTITY", False)
        self.assertNotIn("Canberra", prompt)
        self.assertIn("No answer was accepted.", prompt)


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

    def test_failure_type_retrieval_prefers_the_most_similar_question_within_a_type(self) -> None:
        bank = MemoryBank([
            make_lesson("L1", "bridge", "FORMAT_MISMATCH", question="Which river runs through Vienna?"),
            make_lesson("L2", "bridge", "FORMAT_MISMATCH", question="When was the famous composer born?"),
        ])
        example = Example("q", "When was the Austrian painter born?", "1862", (), qtype="bridge")
        self.assertEqual(Retriever(bank, "failure_type", k=1).select(example)[0].id, "L2")

    def test_retrieval_skips_own_task_and_unvalidated_lessons(self) -> None:
        for mode in ("failure_type", "random", "similarity"):
            ids = {l.id for l in Retriever(self.bank, mode, k=10).select(make_example("target"))}
            self.assertNotIn("L5", ids, mode)
            self.assertNotIn("L6", ids, mode)

    def test_every_mode_returns_the_same_number_of_lessons(self) -> None:
        # Conditions must differ in which lessons are shown, not in how many.
        for example in (make_example("q1"), make_example("q2", qtype="comparison")):
            sizes = {
                mode: len(Retriever(self.bank, mode, k=3).select(example))
                for mode in ("failure_type", "random", "similarity")
            }
            self.assertEqual(set(sizes.values()), {3}, sizes)

    def test_retrieval_is_deterministic_per_seed(self) -> None:
        example = make_example("q1")
        a = Retriever(self.bank, "random", k=2, seed=1).select(example)
        b = Retriever(self.bank, "random", k=2, seed=1).select(example)
        self.assertEqual([l.id for l in a], [l.id for l in b])

    def test_pools(self) -> None:
        self.bank.lessons[-1].fixes_source = False  # L6: usable, but failed its retry
        self.assertEqual(len(self.bank.pool("validated")), 5)
        self.assertEqual(len(self.bank.pool("retry")), 5)
        self.assertEqual(len(self.bank.pool("unvalidated")), 6)
        ids = {l.id for l in Retriever(self.bank, "failure_type", k=10, pool="unvalidated").select(make_example("q"))}
        self.assertIn("L6", ids)
        with self.assertRaises(ValueError):
            self.bank.pool("everything")


class BuildMemoryTests(unittest.TestCase):
    def test_only_lessons_that_fix_the_task_are_validated(self) -> None:
        class StubbornModel(ScriptedProvider):
            def act(self, example, memory, previous_trace, attempt, lessons=()):
                return ActorOutput(thought="t", answer="wrong guess")

        examples = [make_example("never-easy-1"), make_example("never-easy-2")]
        bank, logs = build_memory(examples, StubbornModel())
        self.assertEqual(len(bank.lessons), 2)
        self.assertEqual(bank.validated(), [])
        self.assertTrue(all(l.rejected_reason == REJECT_NO_EFFECT for l in bank.lessons))
        self.assertFalse(any(l.recovered for l in bank.lessons))
        # Three Reflexion attempts, then the fresh retry with only the lesson.
        self.assertEqual([log.trial for log in logs], ["1", "2", "3", "validation"] * 2)

    def test_scripted_lessons_validate(self) -> None:
        bank, logs = build_memory([make_example("zz-hard", answer="Rome")], ScriptedProvider())
        self.assertTrue(all(lesson.validated and lesson.recovered for lesson in bank.lessons))
        # The second attempt was accepted, so there is no third.
        self.assertEqual([log.trial for log in logs], ["1", "2", "validation"])
        self.assertTrue(logs[0].reflection)

    def test_a_lesson_must_also_help_other_questions(self) -> None:
        lesson_questions = [make_example("needs-help"), make_example("harmful-case"), make_example("ok-easy")]
        dev = [make_example("dev-fails"), make_example("ok-fragile-1"), make_example("ok-fragile-2")]
        bank, logs = build_memory(lesson_questions, TaggedModel(), dev_examples=dev)
        good, harmful = bank.lessons
        self.assertEqual([l.id for l in bank.lessons], ["L0001", "L0002"])
        # Both lessons repair the question they came from.
        self.assertTrue(good.fixes_source and harmful.fixes_source)
        # The first fixes the failing development question and breaks nothing.
        self.assertEqual((good.dev_questions, good.dev_helped, good.dev_hurt), (3, 1, 0))
        self.assertTrue(good.validated)
        # The second fixes it too, but breaks two answers that were right.
        self.assertEqual((harmful.dev_helped, harmful.dev_hurt), (1, 2))
        self.assertFalse(harmful.validated)
        self.assertEqual(harmful.rejected_reason, REJECT_NO_TRANSFER)
        self.assertEqual([l.id for l in bank.pool("retry")], ["L0001", "L0002"])
        self.assertEqual(sum(log.trial == "dev_check" for log in logs), 6)
        self.assertEqual(sum(log.trial == "dev_baseline" for log in logs), 3)

    def test_a_lesson_that_names_its_question_is_dropped_before_the_checks(self) -> None:
        class NamingModel(TaggedModel):
            def complete(self, instructions, prompt):
                if instructions == CLASSIFY_INSTRUCTIONS:
                    return "WRONG_ENTITY"
                return "Diagnosis: wrong.\nFix: GOOD rule about Lisbon."

        example = Example("needs-help", "Which city is the capital of Lisbon District?", "Lisbon city", (), qtype="bridge")
        bank, logs = build_memory([example], NamingModel(), dev_examples=[make_example("dev-fails")])
        lesson = bank.lessons[0]
        self.assertEqual(lesson.rejected_reason, REJECT_SPECIFIC)
        self.assertFalse(lesson.usable or lesson.fixes_source or lesson.validated)
        # No retry with the lesson and no development check for it.
        self.assertEqual([log.trial for log in logs], ["1", "2", "dev_baseline"])
        self.assertEqual(bank.pool("unvalidated"), [])

    def test_parallel_build_matches_the_sequential_build(self) -> None:
        lesson_questions = [make_example(f"needs-{i}") for i in range(6)] + [make_example("ok-1")]
        dev = [make_example("dev-fails"), make_example("ok-fragile-1")]
        one, _ = build_memory(lesson_questions, TaggedModel(), dev_examples=dev)
        four, _ = build_memory(lesson_questions, TaggedModel(), dev_examples=dev, workers=4)
        self.assertEqual(one.lessons, four.lessons)

    def test_memory_file_round_trip_keeps_checks(self) -> None:
        bank, _ = build_memory([make_example("needs-help")], TaggedModel(), dev_examples=[make_example("dev-fails")])
        bank.meta = {"model": "tagged"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            bank.save(path)
            loaded = MemoryBank.load(path)
        self.assertEqual(loaded.lessons, bank.lessons)
        self.assertEqual(loaded.meta, {"model": "tagged"})


class ParallelTests(unittest.TestCase):
    def test_results_keep_input_order(self) -> None:
        self.assertEqual(parallel_map(lambda x: x * x, range(50), workers=8), [x * x for x in range(50)])
        seen: list[tuple[int, int]] = []
        parallel_map(lambda x: x, range(5), workers=3, on_done=lambda done, total: seen.append((done, total)))
        self.assertEqual(seen, [(i, 5) for i in range(1, 6)])

    def test_errors_are_raised(self) -> None:
        def fail(value: int) -> int:
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            parallel_map(fail, range(4), workers=2)

    def test_token_budget_goes_into_debt_and_refills(self) -> None:
        budget = TokenBudget(tokens_per_minute=600_000, burst_seconds=0.01)  # 10,000 tokens per second
        budget.spend(300)
        self.assertLess(budget.available, 0)
        budget.wait()  # about 0.02 s until the debt is repaid
        budget._refill()
        self.assertGreater(budget.available, 0)

    def test_usage_is_kept_per_thread(self) -> None:
        provider = _LiveProvider()
        provider._record_usage(10, 5, 0.5)
        seen = {}

        def other_thread() -> None:
            seen["before"] = dict(provider.last_usage)
            provider._record_usage(1, 1, 0.1)
            seen["after"] = dict(provider.last_usage)

        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join()
        self.assertEqual(seen["before"], {})
        self.assertEqual(seen["after"]["prompt_tokens"], 1)
        self.assertEqual(provider.last_usage["prompt_tokens"], 10)
        self.assertEqual(provider.total_usage["calls"], 2)

    def test_retries_wait_and_then_succeed(self) -> None:
        class RateLimited(Exception):
            status_code = 429

        calls = {"n": 0, "retries": 0}

        def flaky() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise RateLimited()
            return "ok"

        result = _with_retries(
            flaky, base_delay=0.001, on_retry=lambda: calls.__setitem__("retries", calls["retries"] + 1)
        )
        self.assertEqual((result, calls["n"], calls["retries"]), ("ok", 3, 2))
        with self.assertRaises(ValueError):
            _with_retries(lambda: (_ for _ in ()).throw(ValueError("not transient")))


class AnalysisTests(unittest.TestCase):
    def test_mcnemar_exact(self) -> None:
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact(0, 8), 2 / 256)
        self.assertAlmostEqual(mcnemar_exact(3, 3), 1.0)

    def test_paired_f1_change_has_an_interval_around_it(self) -> None:
        examples = [make_example(f"e{i}", answer=f"ans{i}") for i in range(12)]
        bank = MemoryBank([make_lesson("L1", "bridge", "WRONG_ENTITY")])
        rows = run_condition("no_memory", examples, ScriptedProvider(), 1, label="rules")
        rows += run_condition("failure_memory", examples, ScriptedProvider(), 1, bank=bank, label="rules")
        change = analyze(rows, bank)["first_attempt_f1_changes"][0]
        failed_before = sum(1 for r in rows if r["condition"] == "no_memory" and not r["exact_match"])
        self.assertEqual(change["condition"], "failure_memory")
        self.assertAlmostEqual(change["f1_change"], failed_before / 12, places=3)
        low, high = change["ci95"]
        self.assertLessEqual(low, change["f1_change"])
        self.assertGreaterEqual(high, change["f1_change"])
        self.assertEqual(paired_f1({}, "no_memory", "reflexion")["pairs"], 0)

    def test_wilson_interval(self) -> None:
        low, high = wilson_interval(65, 100)
        self.assertAlmostEqual(low, 0.5525, places=3)
        self.assertAlmostEqual(high, 0.7364, places=3)
        self.assertEqual(wilson_interval(0, 0), (0.0, 0.0))

    def test_paired_flips_and_lesson_effects(self) -> None:
        examples = [make_example(f"e{i}", answer=f"ans{i}") for i in range(12)]
        bank = MemoryBank([make_lesson("L1", "bridge", "WRONG_ENTITY")])
        model = ScriptedProvider()
        rows = run_condition("no_memory", examples, model, 1, label="rules")
        rows += run_condition("failure_memory", examples, model, 1, bank=bank, label="rules")
        report = analyze(rows, bank)
        comp = next(c for c in report["comparisons"]
                    if c["condition"] == "failure_memory" and c["metric"] == "first")
        failed_before = sum(1 for r in rows if r["condition"] == "no_memory" and not r["exact_match"])
        self.assertEqual(comp["fail_to_pass"], failed_before)
        self.assertEqual(comp["pass_to_fail"], 0)
        self.assertEqual(report["lesson_effects"]["failure_memory:L1"]["helped"], failed_before)
        # Broken down by the failure type the lessons teach, and by question type.
        taught = report["by_lesson_failure_type"]["failure_memory"]["WRONG_ENTITY"]
        self.assertEqual((taught["questions"], taught["fixed"], taught["broken"]), (12, failed_before, 0))
        self.assertEqual(report["by_question_type"]["failure_memory"]["bridge"]["fixed"], failed_before)

    def test_comparison_table_has_one_row_per_condition(self) -> None:
        examples = [make_example(f"e{i}", answer=f"ans{i}") for i in range(9)]
        bank = MemoryBank([make_lesson(f"L{i}", "bridge", "WRONG_ENTITY") for i in range(4)])
        rows = []
        for condition in CONDITIONS:
            rows += run_condition(condition, examples, ScriptedProvider(), 2, bank=bank, label="rules")
        report = analyze(rows, bank)
        table = comparison_table(report)
        self.assertEqual([row["Method"] for row in table], [c.label for c in CONDITIONS.values()])
        by_method = {row["Method"]: row for row in table}
        self.assertEqual(by_method["No memory"]["Lessons"], "0")
        self.assertEqual(by_method["Lessons by failure type (proposed)"]["Lessons"], "3")
        self.assertEqual(by_method["No memory"]["Fixed"], "")  # nothing to compare it with
        self.assertIn("EM after 2 attempts", table[0])

    def test_gain_bootstrap_counts_fixed_minus_broken(self) -> None:
        examples = [make_example(f"e{i}", answer=f"ans{i}") for i in range(12)]
        bank = MemoryBank([make_lesson("L1", "bridge", "WRONG_ENTITY")])
        rows = run_condition("no_memory", examples, ScriptedProvider(), 1, label="rules")
        rows += run_condition("failure_memory", examples, ScriptedProvider(), 1, bank=bank, label="rules")
        report = analyze(rows, bank)
        gain = report["first_attempt_em_gains"][0]
        fixed = sum(1 for r in rows if r["condition"] == "no_memory" and not r["exact_match"])
        self.assertEqual((gain["condition"], gain["pairs"]), ("failure_memory", 12))
        self.assertAlmostEqual(gain["gain_points"], 100 * fixed / 12, places=3)
        low, high = gain["ci95_points"]
        self.assertLessEqual(low, gain["gain_points"])
        self.assertGreaterEqual(high, gain["gain_points"])
        self.assertEqual(sum(gain["net_counts"].values()), gain["resamples"])
        self.assertEqual(gain["share_above_zero"], 1.0)  # the scripted lesson never breaks an answer
        # The same seed gives the same resamples, so a figure can be redrawn exactly.
        self.assertEqual(report["first_attempt_em_gains"], analyze(rows, bank)["first_attempt_em_gains"])
        self.assertEqual(paired_gain({}, "no_memory", "reflexion")["pairs"], 0)

    def test_report_holds_the_curve_of_every_condition(self) -> None:
        examples = [make_example(f"e{i}", answer=f"ans{i}") for i in range(9)]
        rows = run_condition("no_memory", examples, ScriptedProvider(), 3, label="rules")
        rows += run_condition("reflexion", examples, ScriptedProvider(), 3, label="rules")
        curves = analyze(rows)["accuracy_by_attempt"]
        self.assertEqual(len(curves["no_memory"]), 3)
        self.assertEqual(curves["no_memory"][0], curves["no_memory"][-1])  # no memory, no change
        self.assertEqual(curves["reflexion"][0], curves["no_memory"][0])
        self.assertEqual(curves["reflexion"][-1], 1.0)  # the scripted agent recovers after one reflection

    def test_figures_skip_what_a_report_cannot_show(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(plot_all({}, Path(directory)), {})

    def test_table_cells_are_rounded_once(self) -> None:
        # Mean F1 0.82545 and mean response time 0.3452 s. Rounding the report
        # first and the table second would print 0.826 and 0.34.
        rows = [
            {"phase": "eval", "condition": "no_memory", "run_id": 0, "example_id": f"e{i}", "trial": 1,
             "exact_match": 1, "f1": f1, "lesson_ids": "", "n_lessons": 0, "failure_type": "",
             "prompt_tokens": 100, "completion_tokens": 10, "seconds": seconds}
            for i, (f1, seconds) in enumerate([(0.8255, 0.3452), (0.8255, 0.3452), (0.8255, 0.3452), (0.8253, 0.3452)])
        ]
        row = comparison_table(analyze(rows))[0]
        self.assertEqual((row["First attempt F1"], row["Seconds"]), ("0.825", "0.35"))

    def test_repeated_runs_do_not_double_count_questions(self) -> None:
        examples = [make_example(f"e{i}", answer=f"ans{i}") for i in range(6)]
        rows = []
        for run_id in (0, 1):
            rows += run_condition("no_memory", examples, ScriptedProvider(), 1, label="rules", run_id=run_id)
            rows += run_condition("reflexion", examples, ScriptedProvider(), 2, label="rules", run_id=run_id)
        report = analyze(rows)
        self.assertEqual(report["runs"], ["0", "1"])
        self.assertTrue(all(row["examples"] == 6 for row in report["summary"]))
        self.assertTrue(all(comp["pairs"] == 6 for comp in report["comparisons"]))
        self.assertEqual(len(report["by_run"]), 4)

    def test_late_lessons_never_touch_the_first_attempt(self) -> None:
        seen: list[tuple[int, int]] = []

        class Recorder:
            name = "recorder"
            last_usage: dict = {}

            def act(self, example, memory, previous_trace, attempt, lessons=()):
                seen.append((attempt, len(lessons)))
                return ActorOutput("t", "wrong")

            def reflect(self, example, failed_trace, memory):
                return "Try again."

        lessons_for = lambda example: (["a lesson"], ["L1"])  # noqa: E731
        result = run_experiment(
            [make_example("e")], Recorder(), Strategy.REFLEXION, 3,
            lessons_for=lessons_for, lessons_from_trial=2,
        )
        self.assertEqual(seen, [(1, 0), (2, 1), (3, 1)])
        self.assertEqual([record.lesson_ids for record in result.records], [[], ["L1"], ["L1"]])
        self.assertEqual(CONDITIONS["late_memory"].lessons_from, 2)

    def test_a_missing_pool_is_reported(self) -> None:
        bank = MemoryBank([make_lesson("L1", "bridge", "WRONG_ENTITY")])
        bank.lessons[0].fixes_source = False
        with self.assertRaises(ValueError):
            run_condition("retry_memory", [make_example("e")], ScriptedProvider(), 1, bank=bank)


class CliTests(unittest.TestCase):
    def test_offline_pipeline_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = ["--out-dir", directory]
            folder = Path(directory)
            self.assertEqual(experiment_main(["pilot", "--n", "6", *out]), 0)
            self.assertEqual(experiment_main(["build", "--n", "10", "--dev-n", "4", "--workers", "3", *out]), 0)
            bank = MemoryBank.load(folder / "memory.json")
            self.assertEqual(bank.meta["development_questions"], 4)
            self.assertTrue(all(lesson.dev_questions == 4 for lesson in bank.validated()))
            self.assertEqual(
                experiment_main(["evaluate", "--n", "8", "--conditions", "all", "--label", "rules", *out]), 0
            )
            attempts = folder / "eval_attempts.csv"
            self.assertEqual({row["condition"] for row in read_rows(attempts)}, set(CONDITIONS))
            for name in ("eval_report.json", "comparison_table.md", "comparison_table.csv"):
                self.assertTrue((folder / name).exists(), name)
            self.assertEqual(experiment_main(["analyze", "--attempts", str(attempts)]), 0)
            self.assertEqual(
                experiment_main(["analyze", "--attempts", str(attempts), "--layout", "column"]), 0
            )
            if (folder / "comparison.png").exists():  # figures need matplotlib
                for name in ("comparison", "attempts", "fixed_broken", "gain_bootstrap", "lesson_checks"):
                    self.assertTrue((folder / f"{name}.png").exists(), name)
                    self.assertTrue((folder / f"{name}_column.png").exists(), name)
            self.assertEqual(experiment_main(["evaluate", "--split", "validation", "--n", "2", *out]), 1)
            self.assertEqual(experiment_main(["evaluate", "--split", "train", "--n", "2", *out]), 1)

    def test_resume_keeps_finished_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = ["--out-dir", directory, "--label", "rules"]
            attempts = Path(directory) / "eval_attempts.csv"
            self.assertEqual(experiment_main(["evaluate", "--n", "5", "--conditions", "no_memory", *out]), 0)
            before = attempts.read_text(encoding="utf-8")
            self.assertEqual(
                experiment_main(["evaluate", "--n", "5", "--conditions", "no_memory,reflexion", "--resume", *out]), 0
            )
            after = attempts.read_text(encoding="utf-8")
            self.assertTrue(after.startswith(before))
            self.assertEqual({row["condition"] for row in read_rows(attempts)}, {"no_memory", "reflexion"})
            # Other questions in the saved file: refuse to mix them.
            self.assertEqual(experiment_main(["evaluate", "--n", "5", "--seed", "9", "--resume", *out]), 1)


if __name__ == "__main__":
    unittest.main()
