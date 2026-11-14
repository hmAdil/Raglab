# RAGLab: Benchmarking RAG Configurations for Technical Documentation QA

**Course:** UE24AM343AB5, Prompt Engineering and Optimization for Generative AI
**Team 21 | PES University**

---

## 1. Project Summary and Research Objective

Retrieval-Augmented Generation (RAG) systems combine information retrieval with autoregressive language model generation. While many RAG architectures are deployed in practice, system performance depends heavily on design decisions across text segmentation, embedding representations, candidate retrieval, rescoring, and prompt structuring.

RAGLab systematically benchmarks four architectural dimensions on a technical documentation question answering task:
1. **Chunking strategy:** fixed-size, sentence-based, semantic similarity, and recursive character splitting.
2. **Dense embedding model:** BGE-small (`BAAI/bge-small-en-v1.5`) vs. MiniLM (`sentence-transformers/all-MiniLM-L6-v2`).
3. **Retrieval and rescoring setup:** dense-only candidate rescoring vs. hybrid candidate fusion with BM25 using Reciprocal Rank Fusion (RRF), evaluated with both single-vector DistilBERT (`multi-qa-distilbert-cos-v1`) and late-interaction ColBERT (`colbert-ir/colbertv2.0` with MaxSim operator).
4. **Prompt style:** plain instruction-following vs. citation-grounded instruction with strict refusal constraints.

The objective is to measure how these factors interact to influence deterministic retrieval metrics (context recall, context precision, hit rate, latency) and downstream generation quality (answer groundedness, correctness, refusal accuracy, token consumption, and generation latency).

---

## 2. Corpus Description and Question Set

- **Corpus:** 159 technical documentation pages from FastAPI documentation and PyTorch beginner tutorials.
- **Passages:** 1,192 extracted page sections stored in `data/processed/fastapi.jsonl` (712 sections) and `data/processed/pytorch.jsonl` (480 sections).
- **Test Set:** 30 validated questions stored in `data/testset.jsonl`:
  - 27 answerable questions mapped to exact character spans within unique corpus passages (`gold_evidence`).
  - 3 unanswerable questions designed to measure model refusal behavior and hallucination resistance.

---

## 3. Architecture and Pipeline

The end-to-end evaluation pipeline operates in two stages:

```
[Question]
    │
    ├── 1. Embed query with BGE-small or MiniLM
    │      Retrieve top-20 candidates from FAISS exact index (IndexFlatIP)
    │
    ├── 2. (Hybrid setups) Retrieve top-20 candidates via BM25
    │      Union candidates into combined candidate pool
    │
    ├── 3. Rescore candidate pool using ColBERT (MaxSim) or DistilBERT (Cosine)
    │
    ├── 4. Select top-5 chunks:
    │      - Dense-only: Top 5 by neural rescorer score
    │      - Hybrid: Merge neural ranking with BM25 ranking via RRF (k=60), take top 5
    │
    └── 5. Generator (Stage 2):
           - Plain prompt or Grounded prompt
           - LLM Generator (local Ollama llama3.2:3b)
           - LLM Judge (local qwen2.5:7b through Ollama's OpenAI-compatible API)
```

- **Stage 1 (Retrieval Evaluation):** Evaluates all 32 combinations (4 chunkers × 2 embedders × 4 retrieval setups) over the answerable questions using deterministic character-overlap metrics against gold spans. Selects the top 3 configurations.
- **Stage 2 (Generation and Evaluation):** Evaluates the top 3 retrieval configurations with both prompt styles (6 experimental runs × 30 questions = 180 generation and judge calls).

---

## 4. Fixed Experimental Decisions and Constants

All modules read fixed constants from `src/config.py`:

| Constant | Value | Description |
| :--- | :--- | :--- |
| `SEED` | 42 | Random seed for Python, NumPy, and PyTorch operations |
| `TOP_K` | 5 | Chunks returned to generator and scored in Stage 1 |
| `CANDIDATE_DEPTH` | 20 | First-stage FAISS candidate depth and BM25 candidate depth |
| `RRF_K` | 60 | Smoothing constant in Reciprocal Rank Fusion: `1.0 / (k + rank)` |
| `TOP_CONFIGS_FOR_STAGE2` | 3 | Number of top Stage 1 configurations promoted to Stage 2 |
| `TOKENIZER_NAME` | `bert-base-uncased` | Reference tokenizer for chunk length accounting |
| `MAX_CHUNK_TOKENS` | 400 | Maximum allowable BERT tokens per chunk |
| `FIXED_OVERLAP_TOKENS` | 50 | Overlap tokens between adjacent fixed chunks |
| `RECURSIVE_OVERLAP_TOKENS` | 50 | Overlap tokens for recursive character splitter |
| `SENTENCES_PER_CHUNK` | 5 | Sentence grouping window size |
| `SEMANTIC_PERCENTILE` | 20 | Similarity percentile cutoff for semantic splitting |
| `SEMANTIC_SPLIT_MODEL` | `sentence-transformers/all-mpnet-base-v2` | Auxiliary model for sentence embeddings |
| `EMBEDDER_MAX_SEQ` | 512 | Sequence length cap for single-vector encoders |
| `COLBERT_MODEL` | `colbert-ir/colbertv2.0` | Late-interaction multi-vector token model |
| `COLBERT_DOC_TOKENS` | 512 | ColBERT document sequence length cap |
| `COLBERT_QUERY_TOKENS` | 32 | ColBERT query sequence length cap |
| `GEN_NUM_CTX` | 8192 | Ollama generator context window |
| `GEN_MAX_TOKENS` | 400 | Maximum tokens produced by generator |
| `JUDGE_MAX_TOKENS` | 300 | Maximum tokens produced by LLM judge |

---

## 5. Python Environment and Setup

The project specification targets Python 3.10.11.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify test set integrity before running experiments:
```powershell
python src/validate_testset.py
```

---

## 6. Environment Variables

The generator and LLM judge require environment variables. Set them in PowerShell:

```powershell
# Generator settings (Ollama default: localhost:11434 with llama3.2:3b)
$env:GEN_BASE_URL = "http://localhost:11434"
$env:GEN_MODEL = "llama3.2:3b"

# LLM Judge settings (OpenAI-compatible endpoint, stronger model)
# Used for the reported results: local qwen2.5:7b served by Ollama (no real key needed)
$env:JUDGE_BASE_URL = "http://localhost:11434/v1"
$env:JUDGE_API_KEY = "ollama"
$env:JUDGE_MODEL = "qwen2.5:7b"

# To use a hosted judge instead, point the same three variables at any OpenAI-compatible provider:
# $env:JUDGE_BASE_URL = "https://<provider-host>/v1"
# $env:JUDGE_API_KEY = "<your-api-key>"
# $env:JUDGE_MODEL = "<stronger-model-name>"
```

*Note: Never hardcode or commit API keys into source files, git commits, or results files.*

---

## 7. Execution Guide

Commands must be run from the repository root with the virtual environment active.

### Stage 0: Initial Validation
```powershell
python src/validate_testset.py
```

### Stage 1: Build Chunks, Indexes, and Run Deterministic Retrieval Grid
```powershell
# Member 1: Build text chunks and neural indexes
python src/smoke_colbert.py
python src/chunkers.py --chunker all
python tests/check_chunks.py
python src/indexer.py --chunker all --model all
python tests/check_indexes.py

# Member 2: Build BM25 indexes and run Stage 1
python src/retrievers.py --build-bm25 all
python tests/check_metrics.py
python tests/check_retrieval.py
python src/run_stage1.py
python src/charts.py --stage1
```

Stage 1 produces:
- `results/stage1.csv` (32 configuration summaries)
- `results/stage1_per_question.csv` (per-query recall, precision, latency)
- `results/top3.json` (top 3 selected configurations sorted by recall, precision, latency)

### Stage 2: Generation, Judging, and Experimentation
Ensure Ollama is running and both models are downloaded:
```powershell
ollama pull llama3.2:3b
ollama pull qwen2.5:7b
```

The runner generates every answer first, then judges every answer. On a 6 GB GPU the 3B generator and 7B judge cannot be loaded together, and interleaving them made Ollama reload a model on every call, which inflated `generation_ms` by about 4 seconds per answer. Generation therefore runs with the generator loaded and a warm-up call before timing starts.

Run Stage 2:
```powershell
# Smoke test on the first 3 questions
python src/run_stage2.py --limit 3

# Full experiment run (resumes safely if interrupted)
python src/run_stage2.py --resume
```

Stage 2 produces:
- `results/stage2_runs.jsonl` (individual question outputs, token usage, judge assessments)
- `results/stage2.csv` (aggregated metrics per run: groundedness, correctness, refusal accuracy)
- `results/stage2_meta.json` (metadata of model names, date, and constants)

### Stage 3: Judge Validation
```powershell
# Sample 10 random answerable evaluations for blind human scoring
python src/judge_validation.py --make-sheet

# After filling manual_groundedness and manual_correctness (1-5) in results/judge_validation_sheet.csv:
python src/judge_validation.py --report
```

Produces `results/judge_validation.json` containing exact agreement, within-1 agreement, and mean absolute error.

### Stage 4: Decision Table and Analysis
```powershell
python src/analysis.py --decision
python src/charts.py --stage2
```

Produces:
- `results/decision_table.csv` (identifies Best for Accuracy, Best for Speed, and Best Balance)
- `results/charts/` (comparative evaluation PNG charts)

### Stage 5: Launch Demonstration Application
```powershell
python app/gradio_app.py
```
Launches the Gradio interface on `http://127.0.0.1:7860` with Playground, Leaderboard, and Charts tabs.

---

## 8. Artifact and Results Locations

- `artifacts/chunks/`: Generated chunk files (`{chunker}.jsonl`) and summary statistics.
- `artifacts/indexes/`: First-stage FAISS indexes (`index.faiss`) and chunk ID mappings.
- `artifacts/rescoring/`: Pre-computed DistilBERT dense matrices and ColBERT token tensors/offsets.
- `artifacts/bm25/`: Pickled BM25 index objects (`{chunker}.pkl`).
- `cache/`: SHA-256 hashed request/response JSON caches for LLM calls.
- `results/`: Committed CSV, JSON, and JSONL benchmark results.
- `results/charts/`: High-resolution PNG evaluation figures.

---

## 9. Results Summary

**Stage 1 (32 retrieval configurations, 27 answerable questions).** Three configurations tie for the best context recall of 0.741: `recursive__minilm__distilbert`, `recursive__bge__distilbert` and `recursive__bge__distilbert_bm25`, which became the Stage 2 configurations.
- Chunking mattered more than the embedder: mean recall was 0.694 for recursive, 0.676 for fixed, 0.652 for sentence and 0.622 for semantic, while BGE (0.666) and MiniLM (0.656) were close.
- BM25 fusion raised mean recall from 0.624 to 0.698 and roughly doubled retrieval latency (16 ms to 32 ms).
- ColBERT used alone was the weakest rescorer (0.583 against 0.664 for DistilBERT) and became competitive only with BM25 (0.700).
- With 27 questions, one question is worth 0.037 of recall, so near-equal configurations cannot be separated.

**Stage 2 (3 configurations x 2 prompts x 30 questions, judged by qwen2.5:7b).** Groundedness was 4.96 to 5.00 and the judge scored refusal accuracy at 1.00 for all six runs. Reading the answers, all 9 grounded refusals are the clean one-liner "Not found in the provided documents.", while at least one plain answer offered an unrelated 54 percent figure from a PyTorch tutorial and was still scored as a refusal, so the judge's refusal score probably flatters the plain prompt. Correctness ranged from 4.59 to 4.96. The grounded prompt produced about 40 percent shorter answers and 13 to 17 percent faster generation, but lowered correctness for two of three configurations (4.59 against 4.85). The best run on accuracy, speed and balance is `recursive__bge__distilbert_bm25__grounded` (correctness 4.96, total latency 3,519 ms), although the two fastest runs differ by only about 7 ms. Full tables are in `results/stage2.csv` and `results/decision_table.csv`.

---

## 10. Limitations and Reproducibility Notes

1. **Greedy Stage 2 Selection:** Stage 2 tests prompt styles only across the top 3 retrieval setups identified in Stage 1 rather than the full 32 configurations. Potential interactions between prompt style and lower-ranked retrieval configurations are not captured.
2. **Domain-Specific Corpus:** The corpus is restricted to technical documentation for FastAPI and PyTorch. Results may vary on unstructured narrative text or conversational data.
3. **Refusal Granularity:** With 3 unanswerable test questions, refusal accuracy operates in steps of 33.3 percentage points.
4. **Input Length Constraints:** Small embedding models default to smaller sequence windows (MiniLM defaults to 256; raised to 512). Chunks up to 400 tokens fit comfortably without silent truncation.
5. **LLM Judge Biases:** Automated judge scoring exhibits known verbosity and self-enhancement biases. Pointwise 1–5 scoring with strict rubrics and sample manual validation provides empirical quality assurance.
6. **Local Judge:** The judge is a 7B open model (qwen2.5:7b) running locally, which is stronger than the 3B generator and from a different model family, but weaker than hosted frontier judges. Its scores are indicative, and the manual validation sample shows how far they can be trusted.
7. **Latency Measurement:** Generation latency is measured on a single laptop GPU (RTX 3050, 6 GB) with the generator already loaded. Cached LLM calls replay their stored latency by design.
