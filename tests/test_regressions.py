import tempfile
import unittest
from pathlib import Path

from src.countdown import CountdownProblem, build_dataset, compute_reward
from src.experiment_identity import experiment_fingerprint
from scripts.update_rerun_report import (
    START,
    END,
    SUPPLEMENT_START,
    SUPPLEMENT_END,
    replace_with_supplement,
    render_summary,
    update_document,
)


class CountdownRegressionTests(unittest.TestCase):
    def test_reward_requires_integer_operands_and_exact_arithmetic(self):
        problem = CountdownProblem([1, 2, 3], 4, "")
        self.assertEqual(compute_reward("<answer>1.2 / .3</answer>", problem), 0.0)
        self.assertEqual(compute_reward("<answer>(1 + 3) * 2 / 2</answer>", problem), 0.0)
        self.assertEqual(compute_reward("<answer>1 + 3 / 2</answer>", problem), 0.0)
        self.assertEqual(compute_reward("<answer>1 + 2 + 3 - 2</answer>", problem), 0.0)
        self.assertEqual(compute_reward("<answer>(1 + 3) / 2</answer>", CountdownProblem([1, 2, 3], 2, "")), 1.0)
        self.assertEqual(compute_reward("<answer>1/0 + 2 + 3</answer>", problem), 0.0)

    def test_generated_solutions_are_valid(self):
        for problem in build_dataset(200, seed=0):
            self.assertEqual(compute_reward(f"<answer>{problem.solution}</answer>", problem), 1.0)

    def test_training_excludes_permuted_evaluation_problems(self):
        evaluations = build_dataset(64, seed=1000)
        signatures = {(tuple(sorted(p.numbers)), p.target) for p in evaluations}
        train = build_dataset(2000, seed=0, exclude_sigs=signatures)
        self.assertFalse(signatures & {(tuple(sorted(p.numbers)), p.target) for p in train})
        self.assertEqual(train, build_dataset(2000, seed=0, exclude_sigs=signatures))


class ExperimentIdentityTests(unittest.TestCase):
    def test_config_and_code_changes_invalidate_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.py"
            source.write_text("first")
            a = experiment_fingerprint({"seed": 0, "lr": 0.1}, [source])
            self.assertEqual(a, experiment_fingerprint({"lr": 0.1, "seed": 0}, [source]))
            self.assertNotEqual(a, experiment_fingerprint({"seed": 1, "lr": 0.1}, [source]))
            source.write_text("second")
            self.assertNotEqual(a, experiment_fingerprint({"seed": 0, "lr": 0.1}, [source]))


class ReportRegressionTests(unittest.TestCase):
    def test_only_publish_completed_post_update_results(self):
        result = {"method": "best_combo_500", "complete": True,
                  "history": [{"step": 0, "eval_step": 1, "eval_reward": 0.75}]}
        payload = {"status": "running", "args": {"steps": 1}, "results": [result]}
        with self.assertRaises(ValueError):
            render_summary(payload, {})
        payload["status"] = "completed"
        summary = render_summary(payload, {"gpu": "test GPU"})
        self.assertIn("75.0%", summary)
        result["history"][-1]["eval_step"] = 0
        with self.assertRaises(ValueError):
            render_summary(payload, {})

    def test_summary_update_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            path.write_text("# Report\n\nOld content\n")
            summary = "<!-- BEGIN FIXED RERUN RESULTS -->\nnew\n<!-- END FIXED RERUN RESULTS -->"
            update_document(path, summary)
            update_document(path, summary)
            self.assertEqual(path.read_text().count("<!-- BEGIN FIXED RERUN RESULTS -->"), 1)
            self.assertIn("Old content", path.read_text())

    def test_report_moves_rerun_from_header_to_section_eight(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            path.write_text(
                "# Report\n\n"
                f"{START}\n## Old header result\n{END}\n\n"
                "## 一、 Historical section\n"
            )
            supplement = (
                f"{SUPPLEMENT_START}\n"
                "## 八、 修复后补充复跑：Best Combo 500 步\n"
                f"{SUPPLEMENT_END}\n"
            )
            replace_with_supplement(path, supplement)
            replace_with_supplement(path, supplement)
            text = path.read_text()
            self.assertNotIn(START, text)
            self.assertEqual(text.count("## 八、"), 1)
            self.assertEqual(text.count("\n---\n"), 1)
            self.assertGreater(text.index("## 一、"), text.index("# Report"))
            self.assertGreater(text.index("## 八、"), text.index("## 一、"))


if __name__ == "__main__":
    unittest.main()
