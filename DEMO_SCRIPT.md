# RAGLab: Demonstration Script and Presentation Guide

**Course:** UE24AM343AB5, Prompt Engineering and Optimization for Generative AI
**Team 21 | PES University**
**Role:** Member 3 — Generation, Judge, Stage 2 and Demonstration Application

---

## 1. Introduction and Setup (1 minute)

- **Opening Statement:**
  "RAGLab systematically benchmarks four architectural axes in technical documentation question answering: chunking strategies, embedding representations, hybrid neural/lexical retrieval, and prompt styles across 159 pages of FastAPI and PyTorch documentation."
- **Pre-flight Check:**
  - Verify local Ollama service is active: `curl http://localhost:11434/api/tags`
  - Ensure prebuilt index and result files exist under `artifacts/` and `results/`.
  - Launch application:
    ```powershell
    python app/gradio_app.py
    ```
  - Open browser at `http://127.0.0.1:7860`.

---

## 2. Walkthrough Order and Step-by-Step Actions

### Tab 1: Playground (Interactive Demonstration) (3–4 minutes)

1. **Configure Parameters:**
   - **Chunker:** `fixed`
   - **Embedder:** `bge`
   - **Retrieval Setup:** `colbert_bm25` (Demonstrates hybrid lexical + late-interaction neural ranking)
   - **Prompt Style:** `plain`
2. **Submit Technical Question:**
   - Input Question:
     `"What is the purpose of the response_model parameter in a FastAPI path operation decorator?"`
   - Click **Submit**.
3. **Point out Retrieved Chunks:**
   - Review the top 5 chunks displayed in the table.
   - Highlight Rank 1 (`oracle-fastapi-228` or corresponding `fixed` chunk): notice that it directly discusses `response_model` for data documentation, validation, and serialization.
   - Point out scores produced by the Reciprocal Rank Fusion (RRF) between ColBERT and BM25.
4. **Compare Prompt Styles:**
   - Switch **Prompt Style** dropdown from `plain` to `grounded`.
   - Click **Submit** again.
   - Note the differences in the generated answer:
     - The grounded answer includes explicit bracket citations (e.g., `[fixed-000123]`) linking claims to source chunks.
     - Demonstrate hallucination resistance: if an unanswerable question is submitted (e.g. *"How do I design quantum computing circuits in FastAPI?"*), show that the grounded prompt produces the exact refusal string:
       `"Not found in the provided documents."`
5. **Review Execution Metrics:**
   - Point out the metrics line below the answer: retrieval latency (typically < 100 ms), generation latency, prompt token count, and completion token count.

### Tab 2: Leaderboard (Empirical Results) (2 minutes)

1. **Stage 1 Retrieval Grid:**
   - Show the 32-row table evaluating context recall, context precision, and latency.
   - Explain how `recursive` chunking with `colbert_bm25` achieved superior recall over dense-only setups.
2. **Stage 2 Generation Benchmark:**
   - Show the 6-run evaluation comparing plain vs. grounded prompts across the top 3 retrieval setups.
   - Highlight groundedness and correctness trade-offs, noting how citation constraints increase groundedness while slightly decreasing answer length.
3. **Decision Table:**
   - Highlight the three winning configurations:
     - **Best for Accuracy:** Maximum correctness / groundedness.
     - **Best for Speed:** Lowest total latency.
     - **Best Balance:** Optimal trade-off under min-max normalization.

### Tab 3: Charts (Visualizations) (1–2 minutes)

1. Navigate to the **Charts** tab.
2. Walk through key trends:
   - *Recall by Chunker:* Shows how chunk boundaries affect passage coverage.
   - *BM25 vs. Dense:* Demonstrates the recall boost provided by hybrid candidate fusion.
   - *Groundedness vs. Prompt Style:* Visualizes how grounded prompting eliminates unsupported claims.

---

## 3. Explaining the Judge and Validation (1 minute)

- Explain that evaluation uses an automated LLM Judge (OpenAI-compatible) scoring:
  - **Groundedness (1–5):** Are claims supported by the retrieved context?
  - **Correctness (1–5):** Does the answer agree with gold reference answers?
  - **Refusal Accuracy:** Correct identification of unanswerable questions.
- Address Judge Reliability and Biases:
  - Mention known judge biases (verbosity bias, position bias, self-enhancement).
  - Point to `results/judge_validation.json` demonstrating human-judge agreement on a random sample of 10 answerable questions.

---

## 4. Closing Summary (30 seconds)

"In summary, RAGLab demonstrates that hybrid retrieval combined with late-interaction rescoring provides the highest candidate recall, while prompt engineering with citation grounding is essential for eliminating hallucinations and ensuring factual groundedness."

---

## 5. Live Demo Backup Recording Checklist

In the event of network disruption or live service downtime, keep a pre-recorded backup video ready:

- [ ] Screen resolution set to 1920x1080 with 125% browser zoom for readability.
- [ ] Clean browser window with bookmarks hidden and no background tabs.
- [ ] Ollama service pre-warmed so model load latency does not appear in the video.
- [ ] Clip 1 (0:00–1:30): Ask the sample FastAPI question on Playground with `plain` prompt; show answer and chunk table.
- [ ] Clip 2 (1:30–2:30): Switch prompt style to `grounded`; highlight inline citations; ask unanswerable question to show refusal.
- [ ] Clip 3 (2:30–3:30): Switch to Leaderboard tab; scroll through Stage 1, Stage 2, and Decision tables.
- [ ] Clip 4 (3:30–4:30): Switch to Charts tab; highlight the accuracy and latency Pareto frontier.
- [ ] Export video to standard MP4 format named `raglab_demo_backup.mp4` on local drive.
