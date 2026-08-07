# Final CPU Validation

Recorded on 2026-08-08 before the reproduction handoff commit.

## Host scope

- Host: Windows 11, Python 3.13.14 CPU validation venv.
- `nvidia-smi`: unavailable.
- CUDA / Llama-3.1-8B model load / real LLM generation: **NOT RUN**.
- Bash scripts: **NOT EXECUTED ON THIS HOST**. A Windows `bash.exe` launcher exists, but the
  host has no usable Linux `/bin/bash`; do not treat this as syntax or runtime PASS.
- This environment intentionally differs from the frozen Linux environment: Torch 2.7.1,
  NumPy 2.2.6, no vLLM. It is only evidence for non-GPU control flow.

## PASS evidence

1. `python -m unittest discover -s tests -v`: **6 tests passed**.
   Coverage includes mock complete pipeline, official metrics/parsing, dataset/pair audit,
   expected-hash mismatch, successful run artifacts, five-trace schema, failure `error.json`,
   resume/error archival, no-overwrite behavior, and results collection.
2. `python -m compileall -q faithfulrag repro tests`: **PASS**.
3. Import check for all official modules plus runner/preflight/collector: **PASS**.
4. `python -m repro.preflight --output outputs/preflight_cpu.json --json`: **PASS in
   non-strict CPU mode**. All eight configs had zero config/data issues; all configured SHA256
   values matched; MuSiQue/SQuAD negative-golden ID sets and order matched.
5. Eight invocations of `python -m repro.run_experiment --config ... --limit 5 --dry-run`:
   **8/8 PASS**.
6. `python -m repro.cpu_checks`: **PASS** with real
   `sentence-transformers/all-MiniLM-L6-v2` CPU encoding. Five FaithEval samples loaded; first
   sample produced 17 chunks; top cosine scores were approximately 0.651708 and 0.529762.
7. CLI help for `repro.run_experiment`, `repro.preflight`, `repro.cpu_checks`, and
   `repro.collect_results`: **PASS**.
8. `repro.collect_results` against an empty `outputs/results`: **PASS**, wrote valid empty
   `summary.csv` and `summary.md` and reported 0 runs.
9. PowerShell parser for `scripts/setup_cpu_test.ps1`: **PASS**.
10. Secret-value scan over configs/source/scripts/tests/docs: **PASS**. Broad hits were only
    environment-variable names and placeholder instructions. `.gitignore` was verified for
    `.env*`, secrets, credentials keys, caches, model directories/weights, temporary files,
    and generated result runs.
11. `git diff --check`: **PASS** except informational Windows LF-to-CRLF warnings from Git.

## Expected non-strict preflight warnings

- Installed CPU NumPy 2.2.6 differs from formal reproduction NumPy 1.26.4.
- Installed CPU Torch 2.7.1 differs from formal reproduction Torch 2.5.1.
- vLLM is not installed on native Windows.
- NVIDIA GPU is unavailable.

These warnings are expected only for this CPU-validation environment. On the formal GPU host,
`repro.preflight --strict` must return success; otherwise stop.

## Additional audit observations

- Hub revisions observed on 2026-08-08:
  - Llama: `0e9e39f249a16976918f6564b8830bc894c89659`
  - MiniLM: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
  - BGE demo: `d4aa6901d3a41ba39fb536a557fa166f842b0e09`
- Preserved upstream chunking emits an empty first chunk when the first cleaned sentence is
  longer than 20 whitespace words. Trigger counts measured without model inference:
  FaithEval 309/1000, MuSiQue-negative 1099/1772, SQuAD-negative 1092/1769.
- Generated machine-readable local reports are `outputs/cpu_check.json` and
  `outputs/preflight_cpu.json`; they are intentionally ignored because they contain
  machine-specific paths/versions and can be regenerated.
