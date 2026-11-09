import argparse
import csv
from pathlib import Path
import sys
import time
from typing import Any
import gradio as gr
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from config import RESULTS_DIR, CHARTS_DIR
except ImportError:
    ROOT = Path(__file__).resolve().parent.parent
    RESULTS_DIR = ROOT / "results"
    CHARTS_DIR = RESULTS_DIR / "charts"

try:
    from data_access import chunk_index
except ImportError:
    ROOT = Path(__file__).resolve().parent.parent

    def chunk_index(chunker: str) -> dict[str, dict[str, Any]]:
        path = ROOT / "artifacts" / "chunks" / f"{chunker}.jsonl"
        if not path.is_file():
            return {}
        import json
        lookup: dict[str, dict[str, Any]] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    lookup[rec["chunk_id"]] = rec
        return lookup

try:
    from retrievers import get_retriever
except ImportError:
    get_retriever = None

from prompts import format_context
from generator import generate

RETRIEVER_CACHE: dict[tuple[str, str, str], Any] = {}
COLBERT_CACHE_KEYS: list[tuple[str, str, str]] = []
CHUNK_INDEX_CACHE: dict[str, dict[str, Any]] = {}

EXPECTED_CHARTS = [
    ("Stage 1: Recall by Chunker", "stage1_recall_by_chunker.png"),
    ("Stage 1: Recall by Embedder", "stage1_recall_by_embedder.png"),
    ("Stage 1: Recall by Retrieval Setup", "stage1_recall_by_retrieval.png"),
    ("Stage 1: BM25 vs Dense", "stage1_recall_bm25_vs_dense.png"),
    ("Stage 1: Recall vs Latency", "stage1_recall_vs_latency.png"),
    ("Stage 2: Groundedness and Correctness", "stage2_groundedness_correctness.png"),
    ("Stage 2: Correctness vs Latency", "stage2_correctness_vs_latency.png"),
]

def _get_or_create_retriever(chunker: str, embedder: str, retrieval: str) -> Any:
    key = (chunker, embedder, retrieval)
    if key in RETRIEVER_CACHE:
        if "colbert" in retrieval and key in COLBERT_CACHE_KEYS:
            COLBERT_CACHE_KEYS.remove(key)
            COLBERT_CACHE_KEYS.append(key)
        return RETRIEVER_CACHE[key]

    if get_retriever is None:
        raise RuntimeError("Retrievers module (src/retrievers.py) is not yet available.")

    if "colbert" in retrieval:
        if len(COLBERT_CACHE_KEYS) >= 2:
            oldest_key = COLBERT_CACHE_KEYS.pop(0)
            RETRIEVER_CACHE.pop(oldest_key, None)
        COLBERT_CACHE_KEYS.append(key)

    retriever = get_retriever(chunker, embedder, retrieval)
    RETRIEVER_CACHE[key] = retriever
    return retriever

def handle_playground_submit(
    chunker: str,
    embedder: str,
    retrieval: str,
    prompt_style: str,
    question: str,
) -> tuple[list[list[Any]], str, str]:
    q_clean = question.strip()
    if not q_clean:
        return [], "Please enter a question.", "Retrieval: 0.0 ms | Generation: 0.0 ms | Tokens: 0 prompt, 0 completion"

    try:
        retriever = _get_or_create_retriever(chunker, embedder, retrieval)
    except Exception as exc:
        err_msg = f"Retriever initialization error: {type(exc).__name__}: {str(exc)}"
        return [], err_msg, "Retrieval: 0.0 ms | Generation: 0.0 ms"

    t0 = time.perf_counter()
    try:
        hits = retriever.retrieve(q_clean, k=5)
    except Exception as exc:
        err_msg = f"Retrieval failed: {type(exc).__name__}: {str(exc)}"
        return [], err_msg, "Retrieval error"
    retrieval_ms = (time.perf_counter() - t0) * 1000.0

    if chunker not in CHUNK_INDEX_CACHE:
        CHUNK_INDEX_CACHE[chunker] = chunk_index(chunker)
    lookup = CHUNK_INDEX_CACHE[chunker]

    chunks: list[dict[str, Any]] = []
    table_rows: list[list[Any]] = []
    for h in hits:
        rec = lookup.get(h.chunk_id)
        if rec is not None:
            chunks.append(rec)
            text_preview = rec.get("text", "")[:300]
            table_rows.append([
                h.rank,
                h.chunk_id,
                round(float(h.score), 4),
                rec.get("source", ""),
                rec.get("section", ""),
                text_preview,
            ])
        else:
            chunks.append({"chunk_id": h.chunk_id, "text": ""})
            table_rows.append([
                h.rank,
                h.chunk_id,
                round(float(h.score), 4),
                "unknown",
                "unknown",
                "Chunk record not found in artifacts",
            ])

    try:
        gen_res = generate(prompt_style, q_clean, chunks)
        answer = gen_res["answer"]
        gen_ms = float(gen_res["generation_ms"])
        prompt_tokens = int(gen_res["prompt_tokens"])
        completion_tokens = int(gen_res["completion_tokens"])
        metrics_line = (
            f"Retrieval latency: {retrieval_ms:.1f} ms | "
            f"Generation latency: {gen_ms:.1f} ms | "
            f"Prompt tokens: {prompt_tokens} | "
            f"Completion tokens: {completion_tokens}"
        )
        return table_rows, answer, metrics_line
    except Exception as exc:
        error_text = f"Ollama is unavailable or generation failed: {type(exc).__name__}: {str(exc)}"
        metrics_line = f"Retrieval latency: {retrieval_ms:.1f} ms | Generation failed"
        return table_rows, error_text, metrics_line

def _load_csv_dataframe(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        return pd.DataFrame()
    return pd.read_csv(csv_path, encoding="utf-8")

def build_app() -> gr.Blocks:
    with gr.Blocks(title="RAGLab Demonstration Application") as demo:
        gr.Markdown("# RAGLab: Benchmarking RAG Configurations for Technical Documentation QA")

        with gr.Tab("Playground"):
            gr.Markdown("### Interactive Retrieval and Generation")
            with gr.Row():
                chunker_dd = gr.Dropdown(
                    choices=["fixed", "sentence", "semantic", "recursive"],
                    value="fixed",
                    label="Chunker",
                )
                embedder_dd = gr.Dropdown(
                    choices=["bge", "minilm"],
                    value="bge",
                    label="Embedder",
                )
                retrieval_dd = gr.Dropdown(
                    choices=["colbert", "distilbert", "colbert_bm25", "distilbert_bm25"],
                    value="colbert",
                    label="Retrieval Setup",
                )
                style_dd = gr.Dropdown(
                    choices=["plain", "grounded"],
                    value="plain",
                    label="Prompt Style",
                )

            question_tb = gr.Textbox(
                label="Question",
                placeholder="Ask a technical question about FastAPI or PyTorch...",
                lines=2,
            )
            submit_btn = gr.Button("Submit", variant="primary")

            gr.Markdown("#### Retrieved Passages (Top 5)")
            chunks_df = gr.Dataframe(
                headers=["Rank", "Chunk ID", "Score", "Source", "Section", "Text Preview"],
                datatype=["number", "str", "number", "str", "str", "str"],
                label="Retrieved Chunks",
                wrap=True,
            )

            gr.Markdown("#### Generated Answer")
            answer_tb = gr.Textbox(label="Answer", lines=5)
            metrics_tb = gr.Textbox(label="Execution Metrics", interactive=False)

            submit_btn.click(
                fn=handle_playground_submit,
                inputs=[chunker_dd, embedder_dd, retrieval_dd, style_dd, question_tb],
                outputs=[chunks_df, answer_tb, metrics_tb],
            )

        with gr.Tab("Leaderboard"):
            gr.Markdown("### Experimental Leaderboard and Decision Table")

            gr.Markdown("#### Stage 1 Leaderboard (32 Retrieval Configurations)")
            s1_df = _load_csv_dataframe(RESULTS_DIR / "stage1.csv")
            if s1_df.empty:
                gr.Markdown("Stage 1 results file (`results/stage1.csv`) is not yet available.")
            else:
                gr.Dataframe(s1_df, interactive=False)

            gr.Markdown("#### Stage 2 Leaderboard (6 Evaluated Runs)")
            s2_df = _load_csv_dataframe(RESULTS_DIR / "stage2.csv")
            if s2_df.empty:
                gr.Markdown("Stage 2 results file (`results/stage2.csv`) is not yet available.")
            else:
                gr.Dataframe(s2_df, interactive=False)

            gr.Markdown("#### Final Decision Table")
            dec_df = _load_csv_dataframe(RESULTS_DIR / "decision_table.csv")
            if dec_df.empty:
                gr.Markdown("Decision table (`results/decision_table.csv`) is not yet available.")
            else:
                gr.Dataframe(dec_df, interactive=False)

        with gr.Tab("Charts"):
            gr.Markdown("### Experimental Evaluation Charts")
            for title, filename in EXPECTED_CHARTS:
                gr.Markdown(f"#### {title}")
                chart_file = CHARTS_DIR / filename
                if chart_file.is_file():
                    gr.Image(value=str(chart_file), type="filepath", label=title)
                else:
                    gr.Markdown(f"Chart `{filename}` has not been generated yet.")

    return demo

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = build_app()
    app.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )

if __name__ == "__main__":
    main()
