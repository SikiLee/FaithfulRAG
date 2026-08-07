# Project

FaithfulRAG reproduction. Current state: **reproduction engineering is prepared and
CPU-validated; paper-level GPU reproduction has not been executed.**

# Paper

- Title: *FaithfulRAG: Fact-Level Conflict Modeling for Context-Faithful
  Retrieval-Augmented Generation*
- arXiv: <https://arxiv.org/abs/2506.08938>, v1 2025-06-10, current v2 2025-07-08.
- ACL: <https://aclanthology.org/2025.acl-long.1062/>, ACL 2025 Long Papers,
  DOI `10.18653/v1/2025.acl-long.1062`.

# Upstream

- Repo: <https://github.com/XMUDeepLIT/Faithful-RAG>
- Branch/commit: `main` / `9181b1132f2f6548775e4f992a9a44fccdd018e9`
- Tag/release: none; exact paper-time code commit is **UNKNOWN**.

# Current reproduction

- Local branch: `reproduction/llama31-8b`
- Audited handoff tree: `48086302bedf95120a5ff6166c00864eb59f9500`
- Always confirm checkout with: `git rev-parse HEAD && git status --short --branch`

# Completed

- Official repository/paper/code-path/dependency/dataset audit.
- Config-driven runner, checkpoints/resume, logs, exception capture, config/environment/data
  manifests, trace, metrics audit, and result collection.
- Eight frozen configs: paper-oriented HF proxy, Full Context reconstruction, one paper
  ablation reconstruction, one clearly labeled custom ablation, and a GitHub demo variant.
- Formal Linux environment lock, GPU smoke script, CPU validation environment and tests.
- Official source diff and result-sensitivity audit in `OFFICIAL_CODE_MODIFICATIONS.md`.

# Not completed

- Real Meta-Llama-3.1-8B-Instruct model load or generation on an NVIDIA GPU.
- GPU smoke/stability run, full FaithEval/MuSiQue runs, paper-number comparison, or failure
  analysis. Mock success does **not** mean the paper has been reproduced.

# CPU validation

- Unit tests: see `audit/FINAL_CPU_VALIDATION.md` for the final recorded command/result.
- All eight config dry-runs and preflight data/hash checks.
- Real `all-MiniLM-L6-v2` CPU encode/cosine path.
- Mock LLM Self-Fact Mining → alignment → Self-Think → evaluation path.
- Six data files checked for schema/count/hash; negative/golden IDs checked for equality/order.
- Bash/GPU tests: **NOT EXECUTED ON THIS HOST**.

# Frozen data identity

| File | Count | SHA256 |
|---|---:|---|
| `datas/faitheval_data.json` | 1000 | `befa1dcce8cfb49538081f903d78c6df69115c5eed0fa4bd318b9aff6f01ffa3` |
| `datas/musique_negative.json` | 1772 | `a480349414c4d273fb3a7ee7cbc9a87e9ca4f071b95ed057810a885d37eee248` |
| `datas/musique_golden.json` | 1772 | `93818b6bfef75712a55ba918879c399456ec1ea2aa70fbb97b60fab5b52a684a` |
| `datas/squad_negative.json` | 1769 | `2b04c1109bc63b0ee21ef6e1c9e61adcedb45a2a72f6c18f77b419892d6fde0b` |
| `datas/squad_golden.json` | 1769 | `1a86db65c0070b9b332797595014b34fbc664d542c082dfab5e9f60e5ac1579d` |

# Known reproduction risks

- HF public backend is not the paper-declared vLLM stack.
- HF temperature-zero compatibility patch can affect predictions and requires GPU validation.
- Paper MiniLM differs from the public HF demo BGE model.
- Public ACC and two paper wordings disagree.
- No paper-specific code tag; post-paper HF and extraction fixes exist.
- Upstream chunking can emit an empty first chunk; behavior is deliberately preserved.
- Hub model/embedding revisions are observed but not pinned by the public pipeline API.

Full evidence and mitigations: `REPRODUCTION.md` and
`OFFICIAL_CODE_MODIFICATIONS.md`.

# GPU requirements

- Linux x86_64; NVIDIA GPU, recommended at least 24 GB VRAM; driver supporting CUDA 12.4.
- Recommended 64 GB system RAM and at least 60 GB free disk for environment, model cache,
  checkpoints, and outputs. These are conservative engineering recommendations, not paper
  claims.
- Access to `meta-llama/Llama-3.1-8B-Instruct` and
  `sentence-transformers/all-MiniLM-L6-v2`.

# First command after getting GPU

```bash
nvidia-smi
```

Then follow **every** step in `REPRODUCTION.md` section
`FIRST GPU RUN — DO NOT SKIP STEPS`; do not jump directly to a full experiment.

# First experiment

FaithEval-counterfactual, same Llama checkpoint and frozen public-HF proxy settings:

1. Full Context.
2. FaithfulRAG.

Do not start MuSiQue, SQuAD, or ablations until the FaithfulRAG-over-Full-Context trend is
credible.

# Stop condition

If strict preflight/model load/smoke/manual trace/stability fails, or FaithfulRAG does not show
the paper's basic advantage trend over Full Context on FaithEval, stop further GPU experiments
and investigate. The full stop list is in `REPRODUCTION.md`.

# Outputs

Each run stores `status.json`, `trace.json`, `metrics.json`, `predictions.json`,
`raw_predictions.json`, `run.log`, `config.json`, `environment.json`,
`dataset_manifest.json`, intermediates when applicable, and `error.json` on failure.

# Final target

- One backbone: Meta-Llama-3.1-8B-Instruct.
- At least two datasets: FaithEval and MuSiQue-negative.
- Full Context vs FaithfulRAG.
- One or two explicitly classified ablations.
- Official-code metrics, separate metric audit, comparison to paper, and failure analysis.

# Backup

- Copy A: `D:\论文\复现\Faithful-RAG`
- Current `origin` is the authors' public repository, not a confirmed user-owned private
  remote. **Nothing was pushed. Do not push this branch to `origin`.**
- After creating an empty private repository, back up with:

  ```bash
  git remote add private <YOUR_PRIVATE_REPOSITORY_URL>
  git push -u private reproduction/llama31-8b
  ```

- Never commit `.env`, tokens, credentials, model weights, caches, or generated result runs.
