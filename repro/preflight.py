"""Validate a reproduction configuration before reserving a GPU run.

This module deliberately uses the standard library except for optional runtime
introspection (``nltk``).  It never downloads models, datasets, or NLTK data.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SAMPLE_KEYS = ("id", "question", "answer", "context")


def _issue(level: str, code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"level": level, "code": code, "message": message, "details": details}


def _resolve_repo_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_config(config: Any, config_path: Path, repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return schema/data-path issues and a compact configuration summary."""
    issues: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"path": str(config_path)}
    if not isinstance(config, dict):
        return [_issue("error", "config_not_object", "Configuration must be a JSON object.")], summary

    required = (
        "schema_version", "experiment_name", "method", "configuration_class",
        "dataset", "model", "generation", "evaluation", "seed", "output_root",
    )
    for key in required:
        if key not in config:
            issues.append(_issue("error", "config_missing_key", f"Missing required key: {key}", key=key))
    if issues:
        return issues, summary

    if config["schema_version"] != 1:
        issues.append(_issue("error", "schema_version", "Only schema_version 1 is supported.", value=config["schema_version"]))
    for key in ("experiment_name", "method", "configuration_class", "output_root"):
        if not isinstance(config[key], str) or not config[key].strip():
            issues.append(_issue("error", "config_type", f"{key} must be a non-empty string.", key=key))

    dataset = config["dataset"]
    if not isinstance(dataset, dict):
        issues.append(_issue("error", "dataset_not_object", "dataset must be an object."))
    else:
        for key in ("name", "path", "expected_samples"):
            if key not in dataset:
                issues.append(_issue("error", "dataset_missing_key", f"dataset.{key} is required.", key=key))
        if isinstance(dataset.get("path"), str):
            summary["dataset_path"] = str(_resolve_repo_path(dataset["path"], repo_root))
        if not isinstance(dataset.get("expected_samples"), int) or dataset.get("expected_samples", -1) < 0:
            issues.append(_issue("error", "dataset_expected_samples", "dataset.expected_samples must be a non-negative integer."))
        expected_sha256 = dataset.get("expected_sha256")
        if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
            issues.append(_issue("error", "dataset_expected_sha256", "dataset.expected_sha256 must be a 64-character hexadecimal SHA256."))
        if "paired_golden_path" in dataset and not isinstance(dataset["paired_golden_path"], str):
            issues.append(_issue("error", "paired_path_type", "dataset.paired_golden_path must be a string."))
        if "paired_golden_path" in dataset:
            paired_sha256 = dataset.get("paired_golden_sha256")
            if not isinstance(paired_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", paired_sha256):
                issues.append(_issue("error", "paired_expected_sha256", "dataset.paired_golden_sha256 must be a 64-character hexadecimal SHA256."))

    model = config["model"]
    if not isinstance(model, dict) or not isinstance(model.get("backend"), str) or not isinstance(model.get("name"), str):
        issues.append(_issue("error", "model_schema", "model.backend and model.name must be strings."))
    elif not isinstance(model.get("backend_config"), dict):
        issues.append(_issue("error", "model_backend_config", "model.backend_config must be an object."))
    else:
        backend_config = model["backend_config"]
        for key in ("device_map", "torch_dtype", "max_input_tokens", "max_concurrency"):
            if key not in backend_config:
                issues.append(_issue("error", "model_backend_setting", f"model.backend_config.{key} is required.", key=key))
        for key in ("max_input_tokens", "max_concurrency"):
            if key in backend_config and (not isinstance(backend_config[key], int) or backend_config[key] <= 0):
                issues.append(_issue("error", "model_backend_setting", f"model.backend_config.{key} must be a positive integer.", key=key))
        requested_revision = model.get("requested_revision")
        if requested_revision is not None and not isinstance(requested_revision, str):
            issues.append(_issue("error", "model_revision", "model.requested_revision must be null or a string."))
        observed_revision = model.get("observed_revision_2026_08_08")
        if not isinstance(observed_revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", observed_revision):
            issues.append(_issue("error", "model_observed_revision", "model.observed_revision_2026_08_08 must be a 40-character Git SHA."))
    generation = config["generation"]
    if not isinstance(generation, dict):
        issues.append(_issue("error", "generation_schema", "generation must be an object."))
    else:
        for key in ("temperature", "top_p", "top_k", "max_tokens"):
            if key not in generation:
                issues.append(_issue("error", "generation_missing_key", f"generation.{key} is required.", key=key))
        if "max_tokens" in generation and (not isinstance(generation["max_tokens"], int) or generation["max_tokens"] <= 0):
            issues.append(_issue("error", "generation_max_tokens", "generation.max_tokens must be a positive integer."))

    if not isinstance(config.get("seed"), int):
        issues.append(_issue("error", "seed_schema", "seed must be an integer."))
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, dict) or not isinstance(evaluation.get("mode"), str):
        issues.append(_issue("error", "evaluation_schema", "evaluation.mode must be a string."))

    if config.get("method") in {"faithfulrag", "no_self_think"}:
        embedding = config.get("embedding")
        if not isinstance(embedding, dict) or not isinstance(embedding.get("model"), str):
            issues.append(_issue("error", "embedding_schema", "This method requires embedding.model."))
        mining = config.get("mining")
        if not isinstance(mining, dict) or any(key not in mining for key in ("temperature", "top_p", "max_tokens")):
            issues.append(_issue("error", "mining_schema", "This method requires explicit mining temperature, top_p, and max_tokens."))
        alignment = config.get("alignment")
        if not isinstance(alignment, dict):
            issues.append(_issue("error", "alignment_schema", "This method requires alignment settings."))
        else:
            for key in ("sent_topk", "chunk_topk", "chunk_size"):
                if not isinstance(alignment.get(key), int) or alignment[key] <= 0:
                    issues.append(_issue("error", "alignment_setting", f"alignment.{key} must be a positive integer.", key=key))
    return issues, summary


def inspect_dataset(
    path: Path,
    expected_samples: int | None = None,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Inspect the JSON array used by a run and return its IDs for pairing."""
    issues: list[dict[str, Any]] = []
    report: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return report, [_issue("error", "dataset_missing", "Dataset file does not exist.", path=str(path))], []
    report["sha256"] = _sha256(path)
    report["expected_sha256"] = expected_sha256
    if expected_sha256 is not None and report["sha256"].lower() != expected_sha256.lower():
        issues.append(_issue(
            "error",
            "dataset_sha256",
            "Dataset SHA256 differs from config expectation.",
            expected=expected_sha256,
            actual=report["sha256"],
            path=str(path),
        ))
    try:
        data = _read_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return report, [_issue("error", "dataset_json", f"Could not parse dataset JSON: {exc}", path=str(path))], []
    if not isinstance(data, list):
        return report, [_issue("error", "dataset_not_array", "Dataset JSON must be an array.", path=str(path))], []

    report["count"] = len(data)
    report["keys"] = sorted({key for row in data if isinstance(row, dict) for key in row})
    if expected_samples is not None and len(data) != expected_samples:
        issues.append(_issue("error", "dataset_count", "Dataset count differs from config expectation.", expected=expected_samples, actual=len(data)))
    missing_counts: dict[str, int] = {key: 0 for key in REQUIRED_SAMPLE_KEYS}
    non_object = 0
    ids: list[str] = []
    for row in data:
        if not isinstance(row, dict):
            non_object += 1
            continue
        for key in REQUIRED_SAMPLE_KEYS:
            if key not in row or row[key] is None:
                missing_counts[key] += 1
        if row.get("id") is not None:
            ids.append(str(row["id"]))
    report["missing_required_keys"] = {key: count for key, count in missing_counts.items() if count}
    report["non_object_rows"] = non_object
    report["unique_id_count"] = len(set(ids))
    if non_object:
        issues.append(_issue("error", "dataset_non_object_rows", "All dataset entries must be JSON objects.", count=non_object))
    for key, count in missing_counts.items():
        if count:
            issues.append(_issue("error", "dataset_missing_sample_key", f"Dataset samples are missing '{key}'.", key=key, count=count))
    if len(ids) != len(set(ids)):
        issues.append(_issue("error", "dataset_duplicate_id", "Dataset contains duplicate IDs.", count=len(ids) - len(set(ids))))
    return report, issues, ids


def compare_paired_ids(primary_ids: Iterable[str], golden_ids: Iterable[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    primary, golden = list(primary_ids), list(golden_ids)
    primary_set, golden_set = set(primary), set(golden)
    report = {
        "primary_count": len(primary), "golden_count": len(golden),
        "same_id_set": primary_set == golden_set,
        "same_order": primary == golden,
        "only_primary": sorted(primary_set - golden_set)[:10],
        "only_golden": sorted(golden_set - primary_set)[:10],
    }
    issues: list[dict[str, Any]] = []
    if primary_set != golden_set:
        issues.append(_issue("error", "paired_id_set_mismatch", "Negative and golden datasets do not contain the same IDs.", **report))
    elif primary != golden:
        issues.append(_issue("warning", "paired_id_order_mismatch", "Negative and golden datasets have equal ID sets but a different order."))
    return report, issues


def _requested_dependencies(repo_root: Path) -> dict[str, dict[str, str]]:
    """Return package pins grouped by source file (the two locks intentionally differ)."""
    result: dict[str, dict[str, str]] = {}
    for filename in ("requirements.txt", "requirements-repro.txt"):
        path = repo_root / filename
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.match(r"([A-Za-z0-9_.-]+)\s*(.*)", line)
            if match:
                result.setdefault(match.group(1), {})[filename] = match.group(2).strip()
    return result


def inspect_dependencies(repo_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    requested = _requested_dependencies(repo_root)
    installed: dict[str, str | None] = {}
    issues: list[dict[str, Any]] = []
    for package in sorted(requested):
        try:
            installed[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            installed[package] = None
            issues.append(_issue("warning", "dependency_not_installed", "Pinned dependency is not installed in this interpreter.", package=package, requested=requested[package]))
            continue
        # The reproduction lock intentionally takes precedence when it differs
        # from the author's historical requirements (currently NumPy).
        active_pin = requested[package].get("requirements-repro.txt") or requested[package].get("requirements.txt")
        if active_pin and active_pin.startswith("==") and installed[package] != active_pin[2:]:
            issues.append(_issue(
                "warning", "dependency_version_mismatch",
                "Installed dependency differs from the active reproduction pin.",
                package=package, installed=installed[package], expected=active_pin,
            ))
    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "requirements": requested, "installed": installed,
    }, issues


def inspect_nltk() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report: dict[str, Any] = {"installed": importlib.util.find_spec("nltk") is not None, "resources": {}}
    if not report["installed"]:
        return report, [_issue("warning", "nltk_not_installed", "NLTK is not installed; punkt resources were not checked.")]
    try:
        import nltk  # Optional dependency; no download is intentionally attempted.
        for resource in ("punkt", "punkt_tab"):
            try:
                nltk.data.find(f"tokenizers/{resource}")
                report["resources"][resource] = True
            except LookupError:
                report["resources"][resource] = False
    except Exception as exc:  # A broken optional install should be visible, not fatal here.
        return report, [_issue("warning", "nltk_check_failed", f"Could not inspect NLTK resources: {exc}")]
    missing = [name for name, present in report["resources"].items() if not present]
    return report, ([_issue("warning", "nltk_resource_missing", "Required NLTK tokenizer data is missing; run nltk.download for the named resources.", resources=missing)] if missing else [])


def inspect_gpu() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    executable = shutil.which("nvidia-smi")
    report: dict[str, Any] = {"nvidia_smi": executable, "available": False, "gpus": []}
    if not executable:
        return report, [_issue("warning", "gpu_unavailable", "nvidia-smi is not available; this is allowed in non-strict mode.")]
    try:
        completed = subprocess.run(
            [executable, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True, capture_output=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return report, [_issue("warning", "gpu_check_failed", f"nvidia-smi could not be queried: {exc}")]
    if completed.returncode:
        return report, [_issue("warning", "gpu_check_failed", completed.stderr.strip() or "nvidia-smi failed.")]
    report["gpus"] = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    report["available"] = bool(report["gpus"])
    return report, ([] if report["available"] else [_issue("warning", "gpu_unavailable", "No NVIDIA GPU was reported by nvidia-smi.")])


def run_preflight(config_paths: Iterable[str | Path] | None = None, *, repo_root: Path = REPO_ROOT, strict: bool = False) -> dict[str, Any]:
    """Run all non-destructive checks and return a JSON-serializable report."""
    repo_root = Path(repo_root).resolve()
    paths = list(config_paths or sorted((repo_root / "configs" / "reproduction").glob("*.json")))
    report: dict[str, Any] = {"repo_root": str(repo_root), "strict": strict, "configs": [], "issues": []}
    if not paths:
        report["issues"].append(_issue("error", "no_configs", "No reproduction configuration files were selected."))
    inspected_paths: dict[str, tuple[dict[str, Any], list[dict[str, Any]], list[str]]] = {}
    for requested_path in paths:
        config_path = _resolve_repo_path(requested_path, repo_root)
        entry: dict[str, Any] = {"path": str(config_path)}
        if not config_path.is_file():
            entry["issues"] = [_issue("error", "config_missing", "Configuration file does not exist.", path=str(config_path))]
            report["configs"].append(entry)
            report["issues"].extend(entry["issues"])
            continue
        try:
            config = _read_json(config_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            entry["issues"] = [_issue("error", "config_json", f"Could not parse config JSON: {exc}")]
            report["configs"].append(entry)
            report["issues"].extend(entry["issues"])
            continue
        issues, summary = validate_config(config, config_path, repo_root)
        entry.update(summary)
        dataset = config.get("dataset") if isinstance(config, dict) else None
        if isinstance(dataset, dict) and isinstance(dataset.get("path"), str):
            dataset_path = _resolve_repo_path(dataset["path"], repo_root)
            key = str(dataset_path.resolve())
            if key not in inspected_paths:
                inspected_paths[key] = inspect_dataset(
                    dataset_path,
                    dataset.get("expected_samples"),
                    dataset.get("expected_sha256"),
                )
            data_report, data_issues, ids = inspected_paths[key]
            entry["dataset"] = data_report
            issues.extend(data_issues)
            paired = dataset.get("paired_golden_path")
            if isinstance(paired, str):
                paired_path = _resolve_repo_path(paired, repo_root)
                paired_key = str(paired_path.resolve())
                if paired_key not in inspected_paths:
                    inspected_paths[paired_key] = inspect_dataset(
                        paired_path,
                        expected_sha256=dataset.get("paired_golden_sha256"),
                    )
                golden_report, golden_issues, golden_ids = inspected_paths[paired_key]
                entry["paired_golden_dataset"] = golden_report
                issues.extend(golden_issues)
                if not golden_issues and not data_issues:
                    pair_report, pair_issues = compare_paired_ids(ids, golden_ids)
                    entry["paired_id_alignment"] = pair_report
                    issues.extend(pair_issues)
        entry["issues"] = issues
        report["configs"].append(entry)
        report["issues"].extend(issues)
    report["dependencies"], dependency_issues = inspect_dependencies(repo_root)
    report["nltk"], nltk_issues = inspect_nltk()
    report["gpu"], gpu_issues = inspect_gpu()
    report["issues"].extend(dependency_issues + nltk_issues + gpu_issues)
    if strict:
        for issue in report["issues"]:
            if issue["level"] == "warning":
                issue["level"] = "error"
    report["ok"] = not any(issue["level"] == "error" for issue in report["issues"])
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check FaithfulRAG reproduction inputs without loading models.")
    parser.add_argument("--config", action="append", help="Config path; may be specified multiple times. Defaults to all reproduction configs.")
    parser.add_argument("--strict", action="store_true", help="Treat missing optional runtime prerequisites (including GPU) as errors.")
    parser.add_argument("--output", type=Path, help="Also write the full JSON report to this path.")
    parser.add_argument("--json", action="store_true", help="Print the full JSON report instead of a compact status line.")
    args = parser.parse_args(argv)
    report = run_preflight(args.config, strict=args.strict)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"preflight: {'PASS' if report['ok'] else 'FAIL'}; configs={len(report['configs'])}; issues={len(report['issues'])}")
        for issue in report["issues"]:
            print(f"{issue['level'].upper()}: {issue['code']}: {issue['message']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
