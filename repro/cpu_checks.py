"""CPU-only checks for data, tokenization, chunking, embedding, metrics, and I/O."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from datasets import load_dataset

from faithfulrag.evaluate import acc_score, exact_match_score, f1_score
from faithfulrag.modules import ContextualAlignmentModule
from faithfulrag.util import FormatConverter

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--output", default="outputs/cpu_check.json")
    args = parser.parse_args()

    dataset = load_dataset(
        "json", data_files=str(REPO_ROOT / "datas" / "faitheval_data.json"), split="train"
    ).select(range(5))
    config_files = sorted((REPO_ROOT / "configs" / "reproduction").glob("*.json"))
    configs = [json.loads(path.read_text(encoding="utf-8")) for path in config_files]

    import nltk

    nltk.data.find("tokenizers/punkt")
    nltk.data.find("tokenizers/punkt_tab/english")
    sentences = nltk.sent_tokenize(dataset[0]["context"])
    report = {
        "dataset_samples": len(dataset),
        "dataset_columns": dataset.column_names,
        "configs_loaded": len(configs),
        "nltk_sentences_first_sample": len(sentences),
        "metric_sanity": {
            "exact_match": exact_match_score("The Paris", "Paris"),
            "acc_repository_direction": acc_score("Paris", "Paris, France"),
            "f1": f1_score("Paris France", "Paris"),
        },
        "format_sanity": FormatConverter.extract_answer('{"Reason":"test","Answer":"Paris"}'),
    }

    if not args.skip_embedding:
        alignment = ContextualAlignmentModule(args.embedding_model)
        chunks = alignment.chunk_text(dataset[0]["context"], chunk_size=20)
        matches = alignment.calculate_similarity(
            dataset[0]["context"], [dataset[0]["answer"]], top_k=2, chunk_size=20
        )
        report["embedding"] = {
            "model": args.embedding_model,
            "chunks": len(chunks),
            "contains_empty_chunk": "" in chunks,
            "top_matches": matches,
        }

    with tempfile.TemporaryDirectory() as directory:
        test_path = Path(directory) / "roundtrip.json"
        test_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        assert json.loads(test_path.read_text(encoding="utf-8"))["dataset_samples"] == 5
    output = REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
