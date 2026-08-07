import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from repro.collect_results import collect_rows, write_summary
from repro.preflight import compare_paired_ids, inspect_dataset, run_preflight


class PreflightAndSummaryTests(unittest.TestCase):
    def _config(self, path: str, expected_sha256: str, *, paired: str | None = None) -> dict:
        dataset = {
            "name": "tiny",
            "path": path,
            "expected_samples": 2,
            "expected_sha256": expected_sha256,
        }
        if paired:
            dataset["paired_golden_path"] = paired
        return {
            "schema_version": 1, "experiment_name": "tiny", "method": "full_context",
            "configuration_class": "TEST_CONFIG", "dataset": dataset,
            "model": {
                "backend": "hf", "name": "tiny-model", "requested_revision": None,
                "observed_revision_2026_08_08": "1" * 40,
                "backend_config": {
                    "device_map": "cpu", "torch_dtype": "float32",
                    "max_input_tokens": 32, "max_concurrency": 1,
                },
            },
            "generation": {"temperature": 0, "top_p": 1, "top_k": -1, "max_tokens": 2},
            "evaluation": {"mode": "author_repository_metrics_plus_separate_audit"},
            "seed": 42,
            "output_root": "outputs/results",
        }

    def test_dataset_and_pairing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {"id": "a", "question": "q", "answer": "a", "context": "c"},
                {"id": "b", "question": "q", "answer": "a", "context": "c"},
            ]
            (root / "data.json").write_text(json.dumps(rows), encoding="utf-8")
            report, issues, ids = inspect_dataset(root / "data.json", 2)
            self.assertEqual(report["count"], 2)
            self.assertFalse(issues)
            alignment, alignment_issues = compare_paired_ids(ids, list(reversed(ids)))
            self.assertTrue(alignment["same_id_set"])
            self.assertFalse(alignment["same_order"])
            self.assertEqual(alignment_issues[0]["level"], "warning")
            _, hash_issues, _ = inspect_dataset(root / "data.json", 2, "0" * 64)
            self.assertEqual(hash_issues[0]["code"], "dataset_sha256")

    def test_preflight_accepts_valid_local_config_without_gpu_requirement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs" / "reproduction").mkdir(parents=True)
            (root / "datas").mkdir()
            rows = [{"id": str(i), "question": "q", "answer": "a", "context": "c"} for i in range(2)]
            data_path = root / "datas" / "data.json"
            data_path.write_text(json.dumps(rows), encoding="utf-8")
            expected_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
            config_path = root / "configs" / "reproduction" / "tiny.json"
            config_path.write_text(
                json.dumps(self._config("datas/data.json", expected_sha256)),
                encoding="utf-8",
            )
            report = run_preflight([config_path], repo_root=root)
            self.assertTrue(report["ok"], report["issues"])
            self.assertEqual(report["configs"][0]["dataset"]["unique_id_count"], 2)

    def test_collect_results_from_layout_and_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "FaithEval" / "faithfulrag" / "run-1"
            run.mkdir(parents=True)
            (run / "metrics.json").write_text(json.dumps({"accuracy": 0.5, "exact_match": 0.25, "f1": 0.4}), encoding="utf-8")
            (run / "config.json").write_text(json.dumps({"method": "faithfulrag", "dataset": {"name": "FaithEval"}}), encoding="utf-8")
            rows, warnings = collect_rows(root)
            self.assertFalse(warnings)
            self.assertEqual(rows[0]["ACC"], "0.5")
            self.assertEqual(rows[0]["EM"], "0.25")
            csv_path, markdown_path = write_summary(rows, root)
            self.assertIn("Method,Dataset,ACC,EM,F1,MR", csv_path.read_text(encoding="utf-8"))
            self.assertIn("| FaithEval |", markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
