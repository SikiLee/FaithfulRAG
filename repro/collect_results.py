"""Build a small, stable comparison table from run-level ``metrics.json`` files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ("Method", "Dataset", "ACC", "EM", "F1", "MR")
ALIASES = {
    "ACC": ("acc", "accuracy"),
    "EM": ("em", "exact_match", "exactmatch"),
    "F1": ("f1", "f1_score"),
    "MR": ("mr", "memorization_ratio_percent"),
}


def _normalise_key(key: str) -> str:
    return "".join(char for char in key.lower() if char.isalnum())


def _metric_value(payload: Any, aliases: Iterable[str]) -> Any:
    wanted = {_normalise_key(alias) for alias in aliases}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if _normalise_key(str(key)) in wanted and isinstance(value, (int, float)) and not isinstance(value, bool):
                return value
        for value in payload.values():
            found = _metric_value(value, aliases)
            if found is not None:
                return found
    return None


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    for name in ("config.json", "resolved_config.json", "config.resolved.json"):
        path = run_dir / name
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    return value
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                pass
    return {}


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def collect_rows(results_root: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Recursively read metrics files; malformed files are skipped with a reason."""
    rows: list[dict[str, str]] = []
    warnings: list[str] = []
    if not results_root.exists():
        return rows, [f"Results directory does not exist: {results_root}"]
    for metrics_path in sorted(results_root.rglob("metrics.json")):
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            warnings.append(f"Skipped invalid metrics file {metrics_path}: {exc}")
            continue
        if not isinstance(payload, dict):
            warnings.append(f"Skipped non-object metrics file: {metrics_path}")
            continue
        run_dir = metrics_path.parent
        config = _load_run_config(run_dir)
        dataset_config = config.get("dataset") if isinstance(config.get("dataset"), dict) else {}
        method = config.get("method") or payload.get("method")
        dataset = dataset_config.get("name") or payload.get("dataset")
        # Standard output layout: <root>/<dataset>/<method>/<run_id>/metrics.json.
        if not method and len(run_dir.parents) >= 2:
            method = run_dir.parent.name
        if not dataset and len(run_dir.parents) >= 3:
            dataset = run_dir.parent.parent.name
        row = {"Method": str(method or "unknown"), "Dataset": str(dataset or "unknown")}
        for column, aliases in ALIASES.items():
            row[column] = _format_value(_metric_value(payload, aliases))
        row["_source"] = str(metrics_path)
        rows.append(row)
    rows.sort(key=lambda row: (row["Dataset"].lower(), row["Method"].lower(), row["_source"]))
    return rows, warnings


def write_summary(rows: list[dict[str, str]], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path, markdown_path = output_dir / "summary.csv", output_dir / "summary.md"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    visible = tuple(column for column in COLUMNS if column != "MR" or any(row["MR"] for row in rows))
    lines = ["| " + " | ".join(visible) + " |", "| " + " | ".join("---" for _ in visible) + " |"]
    lines.extend("| " + " | ".join(row[column] for column in visible) + " |" for row in rows)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, markdown_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect FaithfulRAG run metrics into CSV and Markdown tables.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "outputs" / "results", help="Results root to scan recursively.")
    parser.add_argument("--output-dir", type=Path, help="Directory for summary.csv and summary.md (default: --root).")
    args = parser.parse_args(argv)
    rows, warnings = collect_rows(args.root)
    csv_path, markdown_path = write_summary(rows, args.output_dir or args.root)
    print(f"Collected {len(rows)} run(s): {csv_path} and {markdown_path}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
