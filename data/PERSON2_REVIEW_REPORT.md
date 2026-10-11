# RAGLab — Person 2 Deliverables Report: Corpus & Data Collection

## 1. Selected Documentation Sources
Per the blueprint recommendation (~100–200 pages total of technical documentation), we selected two complementary technical documentation sources:
1. **FastAPI Official Documentation** (`https://fastapi.tiangolo.com/`)
2. **PyTorch Tutorials & Core Documentation** (`https://docs.pytorch.org/tutorials/`)

Both sources provide clean, high-density technical material with clear section hierarchies, parameter descriptions, conceptual overviews, and code blocks.

---

## 2. Corpus Statistics (For Data Collection Review)

| Metric | FastAPI | PyTorch Tutorials | **Combined Corpus** |
| :--- | :---: | :---: | :---: |
| **Pages Collected** | 80 | 79 | **159 pages** |
| **Passages Extracted** | 712 | 480 | **1,192 passages** |
| **Unique Section Titles** | 635 | 393 | **1,028 sections** |
| **Total Words** | 243,629 | 182,087 | **425,716 words** |
| **Total Tokens (`cl100k_base`)** | 379,479 | 392,901 | **772,380 tokens** |
| **Avg Tokens per Passage** | 533.0 | 818.5 | **648.0 tokens** |

- **Raw HTML storage:** `data/raw/fastapi/` & `data/raw/pytorch/` (cached on disk, gitignored).
- **Cleaned JSONL storage:** `data/processed/fastapi.jsonl` & `data/processed/pytorch.jsonl`.
- **Corpus Summary File:** `data/corpus_stats.json`.

---

## 3. Data Pipeline & Implementation (`src/loaders.py`)

- **Crawling:** Fetches URLs from official `sitemap.xml` feeds, filtered to relevant tutorial and documentation subpaths.
- **Cleaning & Parsing:** Strips headers, footers, sidebars, navigation trees, and script tags using `BeautifulSoup4`.
- **Hierarchy Preservation:** Extracts headings (`h1`, `h2`, `h3`) as `section_title` metadata, keeping associated body text together.
- **Filtering:** Filters out non-informative passages (< 40 characters) and records word and token counts per passage.
- **Special Token Handling:** Handles tokenizer edge cases (e.g. `<|endoftext|>` inside PyTorch ML tutorials) by allowing raw literal tokens during TikToken evaluation.

---

## 4. Test Set Schema & Coordination (`data/TESTSET_FORMAT.md`)

The schema for `data/testset.jsonl` (and individual member files in `data/questions/<member>.jsonl`):

```json
{
  "id": "p2_q01",
  "source_member": "person2",
  "question": "What is the purpose of the response_model parameter in a FastAPI path operation decorator?",
  "reference_answer": "FastAPI uses the response_model parameter to perform data documentation, validation, and to convert and filter the output data to its declared type.",
  "gold_evidence": {
    "source": "fastapi",
    "section": "response_model Parameter",
    "span": "FastAPI will use this response_model to do all the data documentation, validation, etc. and also to convert and filter the output data to its type declaration."
  },
  "answerable": true,
  "type": "factual"
}
```

### Strict Gold Span Rule:
Every answerable question must contain an **exact, character-for-character verbatim quote** from the processed corpus text. This enables deterministic precision/recall evaluation in Stage 1 without needing LLM calls.

---

## 5. Person 2 Test Questions (10 Completed & Validated)

- **Answerable:** 9/10 (90%)
- **Unanswerable:** 1/10 (10%)
- **Distribution:** 3 Factual, 2 Conceptual, 4 Procedural, 1 Unanswerable
- **Validation Status:** `[PASS] All checks passed` via `python src/validate_testset.py`.

| ID | Type | Source | Question Summary | Answerable |
| :--- | :---: | :---: | :--- | :---: |
| `p2_q01` | factual | fastapi | Purpose of `response_model` parameter in path operation | Yes |
| `p2_q02` | factual | fastapi | Query parameters with default values and optionality | Yes |
| `p2_q03` | conceptual | fastapi | Declaring and injecting shared dependencies with `Depends` | Yes |
| `p2_q04` | procedural | fastapi | Initializing OAuth2 password bearer security scheme | Yes |
| `p2_q05` | conceptual | pytorch | What PyTorch tensors are and their role in encoding models | Yes |
| `p2_q06` | procedural | pytorch | GPU availability check and device migration for tensors | Yes |
| `p2_q07` | factual | pytorch | Role of `torch.autograd` differentiation engine in backprop | Yes |
| `p2_q08` | procedural | fastapi | When and how to execute background tasks after response | Yes |
| `p2_q09` | unanswerable | fastapi | Max transformer layers for fictional WebSocket extension | **No** |
| `p2_q10` | procedural | pytorch | Three steps in a PyTorch parameter optimization loop | Yes |

---

## 6. How to Run & Verify

```bash
# 1. View corpus summary statistics
python src/corpus_stats.py

# 2. Merge all question files from data/questions/*.jsonl into data/testset.jsonl
python src/merge_testset.py

# 3. Validate test set schema and verbatim gold span matches
python src/validate_testset.py
```
