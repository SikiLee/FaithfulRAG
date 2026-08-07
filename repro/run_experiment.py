"""Config-driven FaithfulRAG reproduction runner.

This module orchestrates the author's public classes without changing their prompts,
similarity computation, ranking, or official metrics. It adds checkpointing, output
manifests, paper-oriented metric audits, and narrowly defined baselines/ablations.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import random
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import Dataset, load_dataset

from faithfulrag import FaithfulRAG
from faithfulrag.evaluate import f1_score, normalize_answer
from faithfulrag.modules import SelfThinkModule
from faithfulrag.util.logger import logger as core_logger


METHODS = {"faithfulrag", "full_context", "no_fact_mining", "no_self_think"}
VERSION_PACKAGES = (
    "python",
    "torch",
    "transformers",
    "sentence-transformers",
    "datasets",
    "nltk",
    "numpy",
    "vllm",
    "accelerate",
    "huggingface-hub",
)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-").lower()


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def software_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in VERSION_PACKAGES[1:]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def validate_config(config: dict[str, Any]) -> None:
    required = {"method", "dataset", "model", "generation", "seed", "output_root"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")
    if config["method"] not in METHODS:
        raise ValueError(f"Unsupported method {config['method']!r}; choose from {sorted(METHODS)}")
    for key in ("name", "path"):
        if key not in config["dataset"]:
            raise ValueError(f"dataset.{key} is required")
    for key in ("backend", "name", "backend_config"):
        if key not in config["model"]:
            raise ValueError(f"model.{key} is required")
    if config["method"] in {"faithfulrag", "no_self_think"}:
        if not config.get("embedding", {}).get("model"):
            raise ValueError(f"{config['method']} requires embedding.model")
        if not config.get("mining") or not config.get("alignment"):
            raise ValueError(f"{config['method']} requires mining and alignment settings")


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(json.dumps(config))
    if args.backend:
        config["model"]["backend"] = args.backend
    if args.model_name:
        config["model"]["name"] = args.model_name
    if args.embedding_model:
        config.setdefault("embedding", {})["model"] = args.embedding_model
    if args.output_root:
        config["output_root"] = args.output_root
    config["runtime"] = {
        "limit": args.limit,
        "resume": args.resume,
        "trace_index": args.trace_index,
    }
    return config


def setup_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("faithfulrag.reproduction")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    core_logger.addHandler(file_handler)
    return logger


def seed_everything(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        from transformers import set_seed

        set_seed(seed)
    except ImportError:
        pass


def dataset_manifest(
    path: Path,
    dataset: Dataset,
    expected_samples: int | None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    actual_sha256 = sha256(path)
    manifest = {
        "path": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
        "sha256": actual_sha256,
        "expected_sha256": expected_sha256,
        "loaded_samples": len(dataset),
        "columns": list(dataset.column_names),
        "expected_samples": expected_samples,
    }
    if expected_samples is not None and len(dataset) != expected_samples:
        raise ValueError(
            f"Dataset count mismatch for {path}: expected {expected_samples}, found {len(dataset)}"
        )
    if expected_sha256 is not None and actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            f"Dataset SHA256 mismatch for {path}: expected {expected_sha256}, found {actual_sha256}"
        )
    return manifest


def extract_answer(raw: str) -> str:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if isinstance(value, dict):
        key = next((candidate for candidate in value if "answer" in candidate.lower()), None)
        if key is not None:
            return str(value[key])
    return raw


def ground_truths(answer: Any) -> list[str]:
    return [str(value) for value in answer] if isinstance(answer, list) else [str(answer)]


def audited_metrics(dataset: Dataset, predictions: dict[str, str], cot_format: bool) -> dict[str, float]:
    totals = {"exact_normalized": 0.0, "ground_truth_in_prediction": 0.0, "prediction_in_ground_truth": 0.0, "f1": 0.0}
    for item in dataset:
        prediction = predictions.get(str(item["id"]), "")
        if cot_format:
            prediction = extract_answer(prediction)
        normalized_prediction = normalize_answer(str(prediction))
        candidates = [normalize_answer(answer) for answer in ground_truths(item["answer"])]
        totals["exact_normalized"] += max(normalized_prediction == answer for answer in candidates)
        totals["ground_truth_in_prediction"] += max(answer in normalized_prediction for answer in candidates)
        totals["prediction_in_ground_truth"] += max(normalized_prediction in answer for answer in candidates)
        totals["f1"] += max(f1_score(normalized_prediction, answer) for answer in candidates)
    denominator = len(dataset) or 1
    return {key: 100.0 * value / denominator for key, value in totals.items()}


def memorization_ratio(
    dataset: Dataset,
    predictions: dict[str, str],
    paired_golden_path: Path,
    cot_format: bool,
) -> dict[str, float | int | None]:
    golden_rows = load_json(paired_golden_path)
    golden = {str(row["id"]): row["answer"] for row in golden_rows}
    original_matches = 0
    substituted_matches = 0
    for item in dataset:
        item_id = str(item["id"])
        prediction = predictions.get(item_id, "")
        if cot_format:
            prediction = extract_answer(prediction)
        normalized_prediction = normalize_answer(str(prediction))
        original_matches += max(
            normalized_prediction == normalize_answer(answer)
            for answer in ground_truths(golden[item_id])
        )
        substituted_matches += max(
            normalized_prediction == normalize_answer(answer)
            for answer in ground_truths(item["answer"])
        )
    denominator = original_matches + substituted_matches
    return {
        "original_answer_exact_matches_po": original_matches,
        "substituted_answer_exact_matches_ps": substituted_matches,
        "memorization_ratio_percent": 100.0 * original_matches / denominator if denominator else None,
        "formula": "100 * po / (po + ps)",
    }


def record_for_id(records: Any, item_id: str) -> Any:
    if isinstance(records, dict):
        return records.get(item_id)
    if isinstance(records, list):
        return next((record for record in records if str(record.get("id")) == item_id), None)
    return None


async def checkpoint(
    path: Path,
    resume: bool,
    producer: Callable[[], Coroutine[Any, Any, Any]],
    logger: logging.Logger,
) -> Any:
    if resume and path.exists():
        logger.info("Resuming checkpoint %s", path.name)
        return load_json(path)
    value = await producer()
    write_json(path, value)
    return value


def build_self_think(config: dict[str, Any]) -> SelfThinkModule:
    model = config["model"]
    return SelfThinkModule(
        backend_type=model["backend"],
        model_name=model["name"],
        **model["backend_config"],
    )


async def execute(
    config: dict[str, Any],
    dataset: Dataset,
    run_dir: Path,
    resume: bool,
    logger: logging.Logger,
) -> tuple[dict[str, str], dict[str, Any], bool]:
    method = config["method"]
    generation = dict(config["generation"])
    artifacts: dict[str, Any] = {}
    cot_format = config.get("generation_type") in {"normal_cot", "scheduled_cot"}

    if method in {"faithfulrag", "no_self_think"}:
        model = config["model"]
        rag = FaithfulRAG(
            backend_type=model["backend"],
            model_name=model["name"],
            similarity_model=config["embedding"]["model"],
            mining_sampling_params=config["mining"],
            generation_sampling_params=generation,
            **model["backend_config"],
        )
        intermediate = run_dir / "intermediates"
        knowledges = await checkpoint(
            intermediate / "self_knowledges.json",
            resume,
            lambda: rag.fact_mining_module.generate_knowledges(dataset, **config["mining"]),
            logger,
        )
        self_contexts = await checkpoint(
            intermediate / "self_contexts.json",
            resume,
            lambda: rag.fact_mining_module.generate_self_context(
                dataset, knowledges=knowledges, **config["mining"]
            ),
            logger,
        )
        self_facts = await checkpoint(
            intermediate / "self_facts.json",
            resume,
            lambda: rag.fact_mining_module.extract_facts(self_contexts, **config["mining"]),
            logger,
        )
        chunks_path = intermediate / "topk_chunks.json"
        if resume and chunks_path.exists():
            topk_chunks = load_json(chunks_path)
        else:
            alignment = config["alignment"]
            topk_chunks = rag.get_topk_chunks(
                dataset,
                self_facts,
                sent_topk=alignment["sent_topk"],
                chunk_topk=alignment["chunk_topk"],
                chunk_size=alignment["chunk_size"],
            )
            write_json(chunks_path, topk_chunks)
        artifacts.update(
            knowledges=knowledges,
            self_contexts=self_contexts,
            self_facts=self_facts,
            topk_chunks=topk_chunks,
        )

        if method == "faithfulrag":
            predictions = await rag.get_predictions(
                dataset,
                topk_chunks,
                generation_type=config["generation_type"],
                **generation,
            )
        else:
            aligned = {str(row["id"]): row["topk_chunks"] for row in topk_chunks}
            augmented_rows = []
            for item in dataset:
                row = dict(item)
                prefix = " ".join(
                    chunk["chunk"] for chunk in aligned.get(str(item["id"]), [])
                )
                row["context"] = f"{prefix}\n{item.get('context', '')}" if prefix else item.get("context", "")
                augmented_rows.append(row)
            augmented = Dataset.from_list(augmented_rows)
            predictions = await rag.self_think_module.predict_answer_wo_cot(
                augmented, [], **generation
            )
            artifacts["augmented_dataset"] = augmented_rows
            cot_format = False
        return {str(key): value for key, value in predictions.items()}, artifacts, cot_format

    self_think = build_self_think(config)
    if method == "full_context":
        predictions = await self_think.predict_answer_wo_cot(dataset, [], **generation)
        return {str(key): value for key, value in predictions.items()}, artifacts, False
    if method == "no_fact_mining":
        generation["response_format"] = {"type": "json_object"}
        predictions = await self_think.predict_answer_scheduled_cot(dataset, [], **generation)
        return {str(key): value for key, value in predictions.items()}, artifacts, True
    raise AssertionError(f"Unhandled method: {method}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON experiment config")
    parser.add_argument("--limit", type=int, help="Use only the first N samples")
    parser.add_argument("--run-id", help="Stable run directory name; defaults to UTC timestamp")
    parser.add_argument("--output-root", help="Override config output_root")
    parser.add_argument("--backend", choices=("hf", "llamafactory", "openai", "ollama"))
    parser.add_argument("--model-name")
    parser.add_argument("--embedding-model")
    parser.add_argument("--resume", action="store_true", help="Reuse completed intermediate stages")
    parser.add_argument("--trace-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Validate config/data without loading models")
    return parser


async def async_main(args: argparse.Namespace) -> int:
    config_path = resolve_path(args.config)
    config = apply_overrides(load_json(config_path), args)
    validate_config(config)
    dataset_path = resolve_path(config["dataset"]["path"])
    full_dataset = load_dataset("json", data_files=str(dataset_path), split=config["dataset"].get("split", "train"))
    manifest = dataset_manifest(
        dataset_path,
        full_dataset,
        config["dataset"].get("expected_samples"),
        config["dataset"].get("expected_sha256"),
    )
    dataset = full_dataset.select(range(min(args.limit, len(full_dataset)))) if args.limit else full_dataset
    manifest["selected_samples"] = len(dataset)

    if args.dry_run:
        print(json.dumps({"config": config, "dataset": manifest, "status": "dry-run-ok"}, ensure_ascii=False, indent=2))
        return 0

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = resolve_path(config["output_root"])
    run_dir = output_root / slug(config["dataset"]["name"]) / slug(config["method"]) / slug(run_id)
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        print(
            f"Refusing to overwrite non-empty run directory: {run_dir}. "
            "Choose a new --run-id or pass --resume.",
            file=sys.stderr,
        )
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(run_dir)
    config["source_config"] = (
        str(config_path.relative_to(REPO_ROOT))
        if config_path.is_relative_to(REPO_ROOT)
        else str(config_path)
    )
    config["run_id"] = run_id
    config["resolved_output_dir"] = str(run_dir)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "dataset_manifest.json", manifest)
    environment = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "software_versions": software_versions(),
        "git_head": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
        "nvidia_smi": command_output(["nvidia-smi"]),
    }
    write_json(run_dir / "environment.json", environment)
    seed_everything(int(config["seed"]))
    logger.info("Starting %s on %s (%d samples)", config["method"], config["dataset"]["name"], len(dataset))

    try:
        raw_path = run_dir / "raw_predictions.json"
        artifacts: dict[str, Any] = {}
        if args.resume and raw_path.exists():
            logger.info("Resuming raw predictions")
            raw_predictions = load_json(raw_path)
            cot_format = config.get("generation_type") in {"normal_cot", "scheduled_cot"}
            for name in ("self_knowledges", "self_contexts", "self_facts", "topk_chunks"):
                path = run_dir / "intermediates" / f"{name}.json"
                if path.exists():
                    artifacts[name.replace("self_knowledges", "knowledges")] = load_json(path)
        else:
            raw_predictions, artifacts, cot_format = await execute(
                config, dataset, run_dir, args.resume, logger
            )
            write_json(raw_path, raw_predictions)

        official = FaithfulRAG.evaluate(
            None, dataset, raw_predictions, cot_format=cot_format, detailed_output=True
        )
        audit = audited_metrics(dataset, raw_predictions, cot_format)
        metrics: dict[str, Any] = {
            "method": config["method"],
            "dataset": config["dataset"]["name"],
            "num_items": official["num_items"],
            "acc": official["acc"],
            "exact_match": official["exact_match"],
            "f1": official["f1"],
            "primary_metric": "acc (author repository implementation)",
            "official_code_metrics": {key: official[key] for key in ("num_items", "acc", "exact_match", "f1")},
            "paper_wording_audit": audit,
            "metric_warning": "Repository acc tests normalized prediction IN normalized ground truth; paper main text says ground truth IN response, while Appendix C.3 says normalized equality.",
        }
        paired = config["dataset"].get("paired_golden_path")
        if paired:
            metrics["memorization_ratio"] = memorization_ratio(
                dataset, raw_predictions, resolve_path(paired), cot_format
            )
        write_json(run_dir / "metrics.json", metrics)

        detail_by_id = {str(row["id"]): row for row in official["details"]}
        prediction_rows = []
        for item in dataset:
            item_id = str(item["id"])
            detail = detail_by_id[item_id]
            prediction_rows.append(
                {
                    **detail,
                    "raw_prediction": raw_predictions.get(item_id, ""),
                    "parsed_prediction": extract_answer(raw_predictions.get(item_id, "")) if cot_format else raw_predictions.get(item_id, ""),
                }
            )
        write_json(run_dir / "predictions.json", prediction_rows)

        trace_index = min(max(args.trace_index, 0), max(len(dataset) - 1, 0))

        def build_trace(index: int) -> dict[str, Any]:
            trace_item = dict(dataset[index])
            trace_id = str(trace_item.get("id", ""))
            topk_record = record_for_id(artifacts.get("topk_chunks"), trace_id)
            final_record = next(
                (row for row in prediction_rows if str(row["id"]) == trace_id), None
            )
            return {
                "dataset_sample": trace_item,
                "self_knowledge": record_for_id(artifacts.get("knowledges"), trace_id),
                "self_context": record_for_id(artifacts.get("self_contexts"), trace_id),
                "self_facts": record_for_id(artifacts.get("self_facts"), trace_id),
                "topk_chunks": topk_record,
                "self_think_input": {
                    "question": trace_item.get("question"),
                    "choices": trace_item.get("choices"),
                    "original_context": trace_item.get("context"),
                    "aligned_chunks": topk_record,
                    "generation_type": config.get("generation_type"),
                },
                "self_think_output": final_record,
                "final": final_record,
            }

        trace = build_trace(trace_index) if len(dataset) else {}
        trace_indices = list(range(min(5, len(dataset))))
        if trace_indices and trace_index not in trace_indices:
            trace_indices[-1] = trace_index
        trace_samples = [build_trace(index) for index in trace_indices]
        write_json(run_dir / "trace.json", trace)
        write_json(run_dir / "trace_samples.json", trace_samples)
        stale_error = run_dir / "error.json"
        if stale_error.exists():
            archived_error = run_dir / (
                "error.previous."
                + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                + ".json"
            )
            stale_error.replace(archived_error)
        write_json(run_dir / "status.json", {"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        logger.info("Complete. ACC=%.2f EM=%.2f F1=%.2f", metrics["acc"], metrics["exact_match"], metrics["f1"])
        return 0
    except BaseException as error:
        logger.exception("Experiment failed")
        write_json(
            run_dir / "error.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return 1
    finally:
        for handler in list(logger.handlers):
            handler.flush()
            logger.removeHandler(handler)
            if handler in core_logger.handlers:
                core_logger.removeHandler(handler)
            handler.close()


def main() -> int:
    return asyncio.run(async_main(build_arg_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
