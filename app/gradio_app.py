import argparse
import base64
import html
import json
import re
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

STATIC_DIR = Path(__file__).resolve().parent / "static"

CHART_GROUPS = [
    ("Stage 1: retrieval", "stage1_"),
    ("Stage 2: generation", "stage2_"),
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

def _esc(x: Any) -> str:
    return html.escape(str(x), quote=True)


def _hits_html(hits_rows: list[list[Any]]) -> str:
    """Render retrieved passages as animated cards (JS animates .rl-hit and .rl-bar i)."""
    if not hits_rows:
        return '<div class="rl-empty">No passages retrieved yet. Ask a question to run the pipeline.</div>'
    top = max((abs(float(r[2])) for r in hits_rows), default=1.0) or 1.0
    cards = []
    for rank, cid, score, source, section, text in hits_rows:
        width = max(4.0, min(100.0, abs(float(score)) / top * 100.0))
        cards.append(
            '<article class="rl-hit">'
            f'<div class="rl-rank">{_esc(rank)}</div>'
            '<div>'
            '<div class="rl-hit-head">'
            f'<span class="rl-tag src">{_esc(source)}</span>'
            f'<span class="rl-tag">{_esc(section) or "section n/a"}</span>'
            f'<span class="rl-cid">{_esc(cid)}</span>'
            f'<span class="rl-score" data-count="{float(score):.4f}" data-dec="3">0</span>'
            '</div>'
            f'<div class="rl-bar"><i data-w="{width:.1f}"></i></div>'
            f'<p class="rl-text">{_esc(text)}</p>'
            '</div></article>'
        )
    return '<div class="rl-hits">' + "".join(cards) + "</div>"


_METRIC_RE = re.compile(r"^(?P<label>[^:]+):\s*(?P<num>-?\d+(?:\.\d+)?)\s*(?P<unit>ms)?\s*$")


def _metrics_html(text: str) -> str:
    """Turn 'A: 1.2 ms | B: 3' into animated stat chips; plain parts become status chips."""
    chips = []
    for part in [p.strip() for p in (text or "").split("|") if p.strip()]:
        m = _METRIC_RE.match(part)
        if m:
            num = m.group("num")
            dec = len(num.split(".")[1]) if "." in num else 0
            unit = f' data-unit=" {m.group("unit")}"' if m.group("unit") else ""
            chips.append(
                f'<div class="rl-chip">{_esc(m.group("label"))}'
                f'<b data-count="{num}" data-dec="{dec}"{unit}>0</b></div>'
            )
        else:
            err = " err" if re.search(r"fail|error|unavailable", part, re.I) else ""
            chips.append(f'<div class="rl-chip{err}"><b>{_esc(part)}</b></div>')
    return '<div class="rl-chips">' + "".join(chips) + "</div>" if chips else ""


def handle_playground_submit(
    chunker: str,
    embedder: str,
    retrieval: str,
    prompt_style: str,
    question: str,
) -> tuple[str, str, str]:
    """Returns (passage cards HTML, answer text, metrics text)."""
    q_clean = question.strip()
    if not q_clean:
        return _hits_html([]), "Please enter a question.", "Retrieval latency: 0.0 ms | Generation latency: 0.0 ms | Prompt tokens: 0 | Completion tokens: 0"

    try:
        retriever = _get_or_create_retriever(chunker, embedder, retrieval)
    except Exception as exc:
        err_msg = f"Retriever initialization error: {type(exc).__name__}: {str(exc)}"
        return _hits_html([]), err_msg, "Retrieval: 0.0 ms | Generation: 0.0 ms"

    t0 = time.perf_counter()
    try:
        hits = retriever.retrieve(q_clean, k=5)
    except Exception as exc:
        err_msg = f"Retrieval failed: {type(exc).__name__}: {str(exc)}"
        return _hits_html([]), err_msg, "Retrieval error"
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
    hits_html = _hits_html(table_rows)

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
        return hits_html, answer, metrics_line
    except Exception as exc:
        error_text = f"Ollama is unavailable or generation failed: {type(exc).__name__}: {str(exc)}"
        metrics_line = f"Retrieval latency: {retrieval_ms:.1f} ms | Generation failed"
        return hits_html, error_text, metrics_line


def _load_csv_dataframe(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        return pd.DataFrame()
    return pd.read_csv(csv_path, encoding="utf-8")


# ---------------------------------------------------------------- leaderboard
BAR_COLS = {"context_recall", "context_precision", "hit_rate", "groundedness", "correctness",
            "faithfulness", "answer_correctness"}


def _top3_ids() -> set[str]:
    path = RESULTS_DIR / "top3.json"
    if not path.is_file():
        return set()
    try:
        return {r["config_id"] for r in json.loads(path.read_text(encoding="utf-8"))}
    except (ValueError, KeyError, TypeError):
        return set()


def _table_html(df: pd.DataFrame, highlight: set[str] | None = None, sort_col: str | None = None) -> str:
    highlight = highlight or set()
    if sort_col and sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False, kind="stable")
    head = "".join(f"<th>{_esc(c)}</th>" for c in df.columns)
    rows = []
    for _, rec in df.iterrows():
        is_top = str(rec.get("config_id", "")) in highlight
        cells = []
        for col in df.columns:
            v = rec[col]
            if pd.isna(v):
                cells.append('<td data-v="">-</td>')
            elif col in BAR_COLS and isinstance(v, (int, float)) and 0 <= float(v) <= 1:
                cells.append(
                    f'<td data-v="{float(v):.6f}"><div class="rl-cell">'
                    f'<div class="rl-bar"><i data-w="{float(v) * 100:.1f}"></i></div>'
                    f'<span data-count="{float(v):.3f}" data-dec="3">0</span></div></td>'
                )
            elif isinstance(v, float):
                cells.append(f'<td data-v="{v:.6f}">{v:.2f}</td>')
            else:
                badge = '<span class="rl-badge">TOP 3</span>' if (col == "config_id" and is_top) else ""
                cells.append(f'<td data-v="{_esc(v)}">{_esc(v)}{badge}</td>')
        rows.append(f'<tr class="rl-row{" top" if is_top else ""}">{"".join(cells)}</tr>')
    return (f'<div class="rl-tablewrap"><table class="rl-table"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def _section_html(title: str, subtitle: str, df: pd.DataFrame, missing_name: str, **kw: Any) -> str:
    out = f'<div class="rl-sec"><h3>{_esc(title)}</h3><span>{_esc(subtitle)}</span></div>'
    if df.empty:
        return out + (f'<div class="rl-pending"><span class="pulse"></span>'
                      f'<code>{_esc(missing_name)}</code> is not available yet. This fills in automatically once the run finishes.</div>')
    return out + _table_html(df, **kw)


def leaderboard_html() -> str:
    s1 = _load_csv_dataframe(RESULTS_DIR / "stage1.csv")
    s2 = _load_csv_dataframe(RESULTS_DIR / "stage2.csv")
    dec = _load_csv_dataframe(RESULTS_DIR / "decision_table.csv")
    return (
        _section_html("Stage 1 leaderboard", "32 retrieval configurations, click a header to sort",
                      s1, "results/stage1.csv", highlight=_top3_ids(), sort_col="context_recall")
        + _section_html("Stage 2 leaderboard", "top configurations, full generation + judging",
                        s2, "results/stage2.csv", highlight=_top3_ids())
        + _section_html("Final decision table", "the recommended configuration",
                        dec, "results/decision_table.csv", highlight=_top3_ids())
    )


# ---------------------------------------------------------------- charts
def _chart_label(name: str) -> str:
    return name.removesuffix(".png").replace("_", " ").replace("stage1", "Stage 1").replace("stage2", "Stage 2").capitalize()


def charts_html() -> str:
    files = sorted(CHARTS_DIR.glob("*.png")) if CHARTS_DIR.is_dir() else []
    if not files:
        return '<div class="rl-pending"><span class="pulse"></span>No charts generated yet. Run <code>python src/charts.py</code>.</div>'
    out = []
    for title, prefix in CHART_GROUPS:
        group = [f for f in files if f.name.startswith(prefix)]
        if not group:
            out.append(f'<div class="rl-sec"><h3>{_esc(title)}</h3><span>pending</span></div>'
                       '<div class="rl-pending"><span class="pulse"></span>Charts appear here once this stage has finished.</div>')
            continue
        figs = []
        for f in group:
            b64 = base64.b64encode(f.read_bytes()).decode("ascii")
            figs.append(f'<figure class="rl-fig"><img alt="{_esc(_chart_label(f.name))}" src="data:image/png;base64,{b64}">'
                        f'<figcaption>{_esc(_chart_label(f.name))}</figcaption></figure>')
        out.append(f'<div class="rl-sec"><h3>{_esc(title)}</h3><span>{len(group)} charts, click to enlarge</span></div>'
                   f'<div class="rl-gallery">{"".join(figs)}</div>')
    return "".join(out)


# ---------------------------------------------------------------- hero
def _corpus_numbers() -> dict[str, int]:
    nums = {"pages": 159, "passages": 1192}
    stats = RESULTS_DIR.parent / "data" / "corpus_stats.json"
    try:
        comb = json.loads(stats.read_text(encoding="utf-8"))["combined"]
        nums = {"pages": int(comb["total_pages"]), "passages": int(comb["total_passages"])}
    except (OSError, ValueError, KeyError):
        pass
    return nums


def hero_html() -> str:
    n = _corpus_numbers()
    s1 = _load_csv_dataframe(RESULTS_DIR / "stage1.csv")
    configs = len(s1) if not s1.empty else 32
    qs = int(s1["n_questions"].iloc[0]) if not s1.empty and "n_questions" in s1 else 27
    stats = [(configs, "configurations"), (n["passages"], "passages"), (n["pages"], "doc pages"), (qs, "test questions")]
    cells = "".join(f'<div class="rl-stat"><b data-count="{v}">0</b><span>{l}</span></div>' for v, l in stats)
    return (
        '<section class="rl-hero">'
        '<p class="rl-eyebrow">Team 21 &middot; UE24AM343AB5</p>'
        '<h1 class="rl-title">RAGLab</h1>'
        '<p class="rl-sub">Benchmarking retrieval-augmented generation configurations on technical '
        'documentation: FastAPI docs and PyTorch tutorials.</p>'
        f'<div class="rl-stats">{cells}</div></section>'
    )


PIPELINE_HTML = (
    '<div class="rl-pipe">'
    '<div class="rl-node" data-node="chunker"><small>Chunker</small><em>fixed</em></div><span class="rl-arrow">&rarr;</span>'
    '<div class="rl-node" data-node="embedder"><small>Embedder</small><em>bge</em></div><span class="rl-arrow">&rarr;</span>'
    '<div class="rl-node" data-node="retrieval"><small>Retrieval</small><em>Dense</em></div><span class="rl-arrow">&rarr;</span>'
    '<div class="rl-node" data-node="rescore"><small>Rescorer</small><em>ColBERT</em></div><span class="rl-arrow">&rarr;</span>'
    '<div class="rl-node" data-node="generator"><small>Generator</small><em>Ollama</em></div>'
    '</div>'
)


def _assets() -> tuple[str, str]:
    """(css, head html) with anime.js and the UI script inlined, so no static route is needed."""
    css = (STATIC_DIR / "raglab.css").read_text(encoding="utf-8")
    anime_js = (STATIC_DIR / "anime.iife.min.js").read_text(encoding="utf-8")
    ui_js = (STATIC_DIR / "raglab.js").read_text(encoding="utf-8")
    # the skin is dark-only, so force Gradio's dark theme even on a light-mode OS
    force_dark = ("<script>if(!/__theme=/.test(location.search)){var u=new URL(location.href);"
                  "u.searchParams.set('__theme','dark');location.replace(u.href);}</script>")
    head = f"{force_dark}<script>{anime_js}</script><script>{ui_js}</script>"
    return css, head


def build_app() -> gr.Blocks:
    with gr.Blocks(title="RAGLab") as demo:
        gr.HTML(hero_html(), elem_classes="rl-html")

        with gr.Tab("Playground"):
            gr.HTML(PIPELINE_HTML, elem_classes="rl-html")
            with gr.Row():
                # filterable=False makes these pick-from-a-list selects, not type-to-search text boxes
                dd = dict(filterable=False, elem_classes="rl-dd")
                chunker_dd = gr.Dropdown(choices=["fixed", "sentence", "semantic", "recursive"],
                                         value="fixed", label="Chunker", elem_id="rl-dd-chunker", **dd)
                embedder_dd = gr.Dropdown(choices=["bge", "minilm"], value="bge",
                                          label="Embedder", elem_id="rl-dd-embedder", **dd)
                retrieval_dd = gr.Dropdown(choices=["colbert", "distilbert", "colbert_bm25", "distilbert_bm25"],
                                           value="colbert", label="Retrieval Setup", elem_id="rl-dd-retrieval", **dd)
                style_dd = gr.Dropdown(choices=["plain", "grounded"], value="plain",
                                       label="Prompt Style", elem_id="rl-dd-style", **dd)

            question_tb = gr.Textbox(label="Question", elem_id="rl-question", lines=2,
                                     placeholder="Ask a technical question about FastAPI or PyTorch...")
            submit_btn = gr.Button("Run pipeline", variant="primary", elem_id="rl-ask")

            metrics_state = gr.State("")
            metrics_html = gr.HTML("", elem_classes="rl-html", elem_id="rl-metrics")
            answer_md = gr.Markdown("*The generated answer will appear here.*", elem_id="rl-answer")
            gr.HTML('<div class="rl-sec"><h3>Retrieved passages</h3><span>top 5, ranked</span></div>', elem_classes="rl-html")
            hits_html = gr.HTML(_hits_html([]), elem_classes="rl-html", elem_id="rl-hits")

            run_args = dict(
                fn=handle_playground_submit,
                inputs=[chunker_dd, embedder_dd, retrieval_dd, style_dd, question_tb],
                outputs=[hits_html, answer_md, metrics_state],
                show_progress="hidden",
            )
            submit_btn.click(**run_args).then(
                fn=_metrics_html, inputs=metrics_state, outputs=metrics_html, show_progress="hidden")
            question_tb.submit(**run_args).then(
                fn=_metrics_html, inputs=metrics_state, outputs=metrics_html, show_progress="hidden")

        with gr.Tab("Leaderboard"):
            gr.HTML(leaderboard_html(), elem_classes="rl-html")

        with gr.Tab("Charts"):
            gr.HTML(charts_html(), elem_classes="rl-html")

    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    css, head = _assets()
    app = build_app()
    app.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        theme=gr.themes.Base(primary_hue="violet", neutral_hue="slate"),
        css=css,
        head=head,
    )


if __name__ == "__main__":
    main()
