import argparse
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from repro.run_experiment import async_main


class RunnerOutputTest(unittest.TestCase):
    def _make_args(self, root: Path, output_root: Path) -> argparse.Namespace:
        dataset_path = root / "data.json"
        dataset_path.write_text(
            json.dumps([{"id": "one", "question": "Capital?", "answer": "Paris", "context": "Paris is the supplied answer."}]),
            encoding="utf-8",
        )
        config_path = root / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "experiment_name": "io-test",
                    "method": "full_context",
                    "dataset": {"name": "tiny", "path": str(dataset_path), "split": "train", "expected_samples": 1},
                    "model": {"backend": "hf", "name": "unused", "backend_config": {}},
                    "generation_type": "wo_cot",
                    "generation": {"temperature": 0, "top_p": 1, "top_k": -1, "max_tokens": 8},
                    "seed": 42,
                    "output_root": str(output_root),
                }
            ),
            encoding="utf-8",
        )
        return argparse.Namespace(
            config=str(config_path), limit=None, run_id="test-run", output_root=None,
            backend=None, model_name=None, embedding_model=None, resume=False,
            trace_index=0, dry_run=False,
        )

    def test_mocked_generation_writes_complete_run_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_root = root / "results"
            args = self._make_args(root, output_root)
            fake_execute = AsyncMock(return_value=({"one": "Paris"}, {}, False))
            with patch("repro.run_experiment.execute", fake_execute):
                status = asyncio.run(async_main(args))
            self.assertEqual(status, 0)
            run_dir = output_root / "tiny" / "full_context" / "test-run"
            expected = {
                "config.json",
                "dataset_manifest.json",
                "environment.json",
                "raw_predictions.json",
                "predictions.json",
                "metrics.json",
                "trace.json",
                "trace_samples.json",
                "status.json",
                "run.log",
            }
            self.assertTrue(expected.issubset({path.name for path in run_dir.iterdir()}))
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["acc"], 100.0)
            self.assertEqual(metrics["exact_match"], 100.0)
            trace_samples = json.loads(
                (run_dir / "trace_samples.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(trace_samples), 1)
            self.assertIn("self_think_input", trace_samples[0])
            with patch("repro.run_experiment.execute", fake_execute):
                duplicate_status = asyncio.run(async_main(args))
            self.assertEqual(duplicate_status, 2)
            self.assertEqual(fake_execute.await_count, 1)

    def test_failure_writes_error_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_root = root / "results"
            args = self._make_args(root, output_root)
            fake_execute = AsyncMock(side_effect=RuntimeError("synthetic failure"))
            with patch("repro.run_experiment.execute", fake_execute):
                status = asyncio.run(async_main(args))
            self.assertEqual(status, 1)
            run_dir = output_root / "tiny" / "full_context" / "test-run"
            error = json.loads((run_dir / "error.json").read_text(encoding="utf-8"))
            self.assertEqual(error["type"], "RuntimeError")
            self.assertIn("synthetic failure", error["message"])
            args.resume = True
            with patch(
                "repro.run_experiment.execute",
                AsyncMock(return_value=({"one": "Paris"}, {}, False)),
            ):
                resumed_status = asyncio.run(async_main(args))
            self.assertEqual(resumed_status, 0)
            self.assertFalse((run_dir / "error.json").exists())
            self.assertEqual(len(list(run_dir.glob("error.previous.*.json"))), 1)


if __name__ == "__main__":
    unittest.main()
