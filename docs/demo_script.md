# RAGLab demo script (about 4 minutes)

Every answer quoted below was produced by the app on 14 November 2026 with llama3.2:3b at temperature 0 and seed 42, so it should repeat exactly. Test-set questions are also stored in the LLM cache, so they return instantly and replay their stored latency.

## Before you present

1. Start Ollama and confirm the generator exists: `ollama list` should show `llama3.2:3b`. The judge is not needed for the demo.
2. From the repository root with the virtual environment active: `python app/gradio_app.py`, then open `http://127.0.0.1:7860`.
3. Close other GPU programs. The 6 GB card needs the memory.
4. Warm up once: ask any question with each configuration you will use, because the first call loads the retriever (ColBERT takes a few seconds).
5. Keep the Leaderboard and Charts tabs open in a second browser tab as a fallback. They work without Ollama.

## Walkthrough

### 1. Leaderboard (45 seconds)

- Stage 1 table: 32 retrieval configurations. The top 3 are highlighted. All three tie at 0.741 context recall, all use recursive chunking and DistilBERT.
- Stage 2 table and decision table: six runs, and the same run wins all three criteria (recursive, BGE, DistilBERT with BM25, grounded prompt). Say that the speed gap between the top two runs is only about 7 ms.

### 2. Playground: refusing when the answer is not there (90 seconds)

Set chunker `recursive`, embedder `bge`, retrieval `distilbert_bm25`. Ask:

> What exact test accuracy did the RAGLab team's private PyTorch model achieve in its latest internal training run?

- Plain prompt: says the exact value is not stated, then adds that the model "achieved an accuracy of 54%". That figure comes from a PyTorch tutorial, not from any RAGLab model.
- Grounded prompt: `Not found in the provided documents.`

Point out that our judge scored both answers as refusals, which is one of the limits of an automated judge.

### 3. Playground: citations (60 seconds)

Set chunker `fixed`, embedder `minilm`, retrieval `colbert`, grounded prompt. Ask:

> How do you make a query parameter required in a FastAPI path operation function?

The answer starts with `[fixed-000150]`, the chunk it used. Find that chunk in the retrieved passages list on the page. Switch to `plain` to show the same answer without a citation.

### 4. Charts (30 seconds)

Show the recall by chunker and recall by retrieval charts. The takeaway is that chunking and BM25 mattered, the embedder barely did, and ColBERT alone was weakest.

## What to say if asked

- Why is correctness lower with the grounded prompt in two configurations? In the three answers we read, the 3B model sometimes answered "not found" or left out a detail. We did not test the cause.
- Why a 7B local judge? It fits the laptop, needs no API key, and is a different model family from the generator. The cost is a lenient judge, which is why we validated it on 10 answers by hand.
- Only 30 questions. Differences of a few points are noise, and we say so in the report.

## If something fails

- Ollama not running: the Playground shows an error in the answer box instead of crashing. Use the Leaderboard and Charts tabs, which only read saved results.
- Slow first answer: the retriever or generator is loading. Wait about 10 seconds and do not click again.
- Total failure: use the screen recording or the saved charts in `results/charts/`.
