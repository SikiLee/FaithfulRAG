# Official Code Modifications Audit

This file is the audit boundary between the authors' public implementation and the
reproduction engineering added in this branch.  Risk means **risk of changing paper
predictions or metrics**, not a security rating.

## Upstream

- Repository: <https://github.com/XMUDeepLIT/Faithful-RAG>
- Remote: `origin`
- Branch: `main`
- Commit: `9181b1132f2f6548775e4f992a9a44fccdd018e9`
- Tag/release: none in the Git repository; no paper-specific code tag was found.
- Exact paper-time source commit: **UNKNOWN**. The public history does not identify it.
- Recreate the complete official-source diff with:

  ```bash
  git diff 9181b1132f2f6548775e4f992a9a44fccdd018e9 -- faithfulrag/
  ```

## Current reproduction

- Branch: `reproduction/llama31-8b`
- Audited handoff tree: `HANDOFF_FINALIZED_AT_COMMIT_PENDING`
- Resolve the checked-out commit: `git rev-parse HEAD`
- Reproduction-only code lives under `repro/`, `scripts/`, `configs/reproduction/`,
  `tests/`, and the top-level reproduction documents/locks.

## Modified official files

### File: `faithfulrag/llm/backend.py`

- Original behavior: imports `logger` as a top-level `util` package and always launches up
  to 10 asynchronous generations, including 10 concurrent calls into one in-process HF
  model.
- Current change: package-relative logger import; consumes `max_concurrency`; formal HF
  configs set it to 1 while non-HF defaults remain 10.
- Reason: importing `faithfulrag` as a normal package otherwise resolves `util`
  inconsistently; concurrent threads over one 8B model can multiply activation memory and
  make a 24 GB first run unstable.
- Type: packaging/import compatibility; device/runtime compatibility.
- Prediction impact: relative import none expected. Sequential HF scheduling should not
  change greedy decoding, but GPU kernels and shared-state behavior have not been verified.
- Metric impact: none expected, but possible indirectly if scheduling changes a prediction.
- Pure engineering fix: import yes; concurrency control yes, with runtime caveat.
- Wrapper alternative: import can only be avoided with fragile `sys.path` manipulation;
  concurrency could be monkey-patched, which is less auditable than one explicit option.
- Risk: **MEDIUM** (highest sub-change risk).
- Recommendation: **KEEP**, and compare a small GPU trace before a full run.

### File: `faithfulrag/llm/hf.py`

- Original behavior: top-level `util` import; always initializes the model with function
  defaults; always sets `do_sample=True`; passes pipeline `temperature=0.0`, `top_p`, and
  OpenAI-style `response_format` through to Transformers; hard-codes input length 8192.
- Current change: package-relative import; consumes explicit device/dtype/quantization/input
  settings; maps `temperature=0` to `do_sample=False`; applies temperature/top-p only while
  sampling; removes non-Transformers kwargs; makes the existing 8192 limit configurable.
- Reason: Transformers 4.49.0's `TemperatureLogitsWarper(0.0)` deterministically raises a
  `ValueError` and instructs callers to use `do_sample=False`. `response_format` would also
  reach `model.generate` as an unsupported kwarg. The paper and author modules both specify
  temperature 0.
- Type: packaging/import compatibility; device/runtime compatibility; generation behavior.
- Prediction impact: **YES**. Greedy decoding is the intended executable interpretation of
  temperature 0, but it is not byte-for-byte the upstream HF code (which crashes for this
  configuration), and HF may differ from the paper's vLLM serving path.
- Metric impact: **YES, potentially**, through generated predictions.
- Pure engineering fix: import/kwarg filtering/device plumbing are engineering fixes;
  decoding mode is a necessary but result-sensitive compatibility decision.
- Wrapper alternative: a separate copied HF backend could avoid this source diff but would
  duplicate more official code and obscure which backend actually ran.
- Risk: **HIGH**.
- Recommendation: **KEEP**, do not call it paper-identical, and gate full experiments on
  manual smoke traces plus the FaithEval Full Context vs FaithfulRAG trend.

### File: `faithfulrag/llm/llamafactory.py`

- Original behavior: top-level `util` logger import.
- Current change: package-relative logger import only.
- Reason: package import compatibility.
- Type: packaging/import compatibility.
- Prediction impact: no.
- Metric impact: no.
- Pure engineering fix: yes.
- Wrapper alternative: fragile `sys.path` injection only.
- Risk: **LOW**.
- Recommendation: **KEEP**.

### File: `faithfulrag/llm/ollama.py`

- Original behavior: top-level `util` logger import.
- Current change: package-relative logger import only.
- Reason: package import compatibility.
- Type: packaging/import compatibility.
- Prediction impact: no.
- Metric impact: no.
- Pure engineering fix: yes.
- Wrapper alternative: fragile `sys.path` injection only.
- Risk: **LOW**.
- Recommendation: **KEEP**.

### File: `faithfulrag/modules.py`

- Original behavior: top-level `util` logger import.
- Current change: package-relative logger import only.
- Reason: package import compatibility.
- Type: packaging/import compatibility.
- Prediction impact: no.
- Metric impact: no.
- Pure engineering fix: yes.
- Wrapper alternative: fragile `sys.path` injection only.
- Risk: **LOW**.
- Recommendation: **KEEP**.
- Important non-change: chunking, fact extraction, similarity, top-k, prompts, and regexes are
  exactly upstream at the pinned commit. An earlier local empty-chunk fix was deliberately
  reverted during final audit because it could alter alignment.

### File: `faithfulrag/util/format_util.py`

- Original behavior: imports `datasets`, `vllm`, `torch`, and other modules that are never
  referenced by `FormatConverter`; any package import therefore requires vLLM.
- Current change: removes only unused imports; converter functions are unchanged.
- Reason: vLLM is unavailable on native Windows and is not used by the public HF pipeline;
  unused imports prevented CPU validation of otherwise independent code.
- Type: dependency compatibility.
- Prediction impact: no expected code path change.
- Metric impact: no.
- Pure engineering fix: yes.
- Wrapper alternative: install/stub vLLM or dynamically inject fake modules; both are less
  faithful and harder to audit.
- Risk: **LOW**.
- Recommendation: **KEEP**.

## Explicitly unmodified official behavior

- `faithfulrag/prompts.py`: no local diff.
- `faithfulrag/evaluate.py`: no local diff; official ACC is preserved.
- Self-Fact extraction prompt and regex: no local diff from upstream commit.
- Context cleaning/chunking algorithm: no local diff.
- SentenceTransformer encoding and `util.cos_sim`: no local diff.
- Per-fact `sent_topk` and global unique `chunk_topk`: no local diff.
- Self-Think scheduled prompt and final prompt assembly: no local diff.

## Known paper-code discrepancies

1. **Embedding model.** Paper Appendix B.5 reports `all-MiniLM-L6-v2`; the public HF demo
   uses `BAAI/bge-large-en-v1.5`. Formal configs use MiniLM and preserve a separately labeled
   demo config.
2. **Backend.** Paper Appendix C.3 reports vLLM. Public `LLMBackend.BACKENDS` contains HF,
   OpenAI, LLaMA Factory, and Ollama, but no vLLM implementation. The formal runnable config
   is therefore a public-HF proxy, not a serving-stack-identical reproduction.
3. **Dependency conflict.** Upstream pins `numpy==2.2.6`; published metadata for
   `vllm==0.6.4.post1` requires `numpy<2.0.0` and `torch==2.5.1`. The reproduction lock uses
   NumPy 1.26.4 and records that deviation.
4. **ACC definitions.** Public code evaluates
   `normalize(prediction) in normalize(ground_truth)`. Paper main text describes the response
   containing the ground truth; Appendix C.3 describes normalized equality. The runner keeps
   public ACC at the top level and writes the other interpretations only under
   `paper_wording_audit`.
5. **Negative/golden semantics.** Negative-file answers are the substituted, context-faithful
   targets; golden answers are original-world targets used for comparison/MR. They are not
   interchangeable.
6. **Paper-time commit.** The arXiv submission is dated 2025-06-10 and the paper appears at
   ACL 2025. Public commits `dd21b2a` (2025-09-01) and `ab7e75e` (2025-09-08) later fix HF
   `max_tokens` forwarding and the Self-Fact extraction prompt call. Whether the paper tables
   used those exact fixes is **UNKNOWN** because there is no paper code tag.
7. **Self-Fact extraction.** Current upstream uses
   `generate_context_extract(user_context=ctx['context'])`. Parent commit of `ab7e75e` used
   the self-knowledge prompt on the generated context. This reproduction does not add another
   local extraction change; result sensitivity to the post-paper upstream fix is **HIGH**.
8. **Chunking edge case.** Upstream appends an empty first chunk whenever the cleaned first
   sentence exceeds 20 whitespace words. The current data trigger this for 309/1000 FaithEval,
   1099/1772 MuSiQue-negative, and 1092/1769 SQuAD-negative samples. It is preserved for public
   code fidelity, but may differ from the paper's described fixed-size chunking or hidden
   runner. Detect it by inspecting `trace.json`/`topk_chunks` and intermediate chunks.
9. **Model revisions.** The public API does not pin Hugging Face revisions. Configs record
   `requested_revision: null` plus Hub revisions observed on 2026-08-08; the revision actually
   downloaded later can change unless the pipeline is extended after author confirmation.
