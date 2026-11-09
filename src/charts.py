import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analysis import load_stage1, load_stage2
from config import CHARTS_DIR

DPI = 150


def save(fig, name: str) -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / name, dpi=DPI)
    plt.close(fig)


def bar_mean(df: pd.DataFrame, by: str, title: str, xlabel: str, name: str) -> None:
    means = df.groupby(by)["context_recall"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(i) for i in means.index], means.values, color="#4C78A8")
    for i, v in enumerate(means.values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean context recall")
    ax.set_ylim(0, 1)
    save(fig, name)


def stage1_charts() -> None:
    df = load_stage1().copy()
    bar_mean(df, "chunker", "Mean context recall by chunker", "Chunker", "stage1_recall_by_chunker.png")
    bar_mean(df, "embedder", "Mean context recall by embedder", "Embedder", "stage1_recall_by_embedder.png")
    bar_mean(df, "retrieval", "Mean context recall by retrieval setup", "Retrieval setup",
             "stage1_recall_by_retrieval.png")
    df["bm25"] = df["retrieval"].map(lambda r: "with BM25" if r.endswith("_bm25") else "without BM25")
    bar_mean(df, "bm25", "Mean context recall with and without BM25", "Retrieval type",
             "stage1_recall_bm25.png")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(df["retrieval_latency_ms"], df["context_recall"], color="#4C78A8")
    for r in df.itertuples():
        ax.annotate(r.config_id, (r.retrieval_latency_ms, r.context_recall), fontsize=5,
                    xytext=(2, 2), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_title("Context recall versus retrieval latency (Stage 1)")
    ax.set_xlabel("Retrieval latency per query (ms, log scale)")
    ax.set_ylabel("Context recall")
    save(fig, "stage1_recall_vs_latency.png")
    print(f"wrote 5 stage 1 charts to {CHARTS_DIR}")


def stage2_charts() -> None:
    df = load_stage2().copy()
    df["total_latency_ms"] = df["retrieval_latency_ms"] + df["generation_latency_ms"]
    configs = list(dict.fromkeys(df["config_id"]))
    for metric, label in (("groundedness", "Groundedness"), ("correctness", "Correctness")):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        width = 0.38
        for j, style in enumerate(("plain", "grounded")):
            vals = [df[(df["config_id"] == c) & (df["prompt_style"] == style)][metric].mean()
                    for c in configs]
            xs = [i + (j - 0.5) * width for i in range(len(configs))]
            ax.bar(xs, vals, width, label=style)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=15, ha="right", fontsize=7)
        ax.set_title(f"{label} by prompt style for the top configurations")
        ax.set_xlabel("Configuration")
        ax.set_ylabel(f"Mean {label.lower()} (1 to 5)")
        ax.set_ylim(0, 5)
        ax.legend(title="Prompt style")
        save(fig, f"stage2_{metric}_by_prompt.png")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(df["total_latency_ms"], df["correctness"], color="#F58518")
    for r in df.itertuples():
        ax.annotate(r.run_id, (r.total_latency_ms, r.correctness), fontsize=6,
                    xytext=(2, 2), textcoords="offset points")
    ax.set_title("Correctness versus total latency (Stage 2)")
    ax.set_xlabel("Total latency per question (ms)")
    ax.set_ylabel("Mean correctness (1 to 5)")
    save(fig, "stage2_correctness_vs_latency.png")
    print(f"wrote 3 stage 2 charts to {CHARTS_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", action="store_true")
    parser.add_argument("--stage2", action="store_true")
    args = parser.parse_args()
    if not (args.stage1 or args.stage2):
        parser.error("pass --stage1 and/or --stage2")
    if args.stage1:
        stage1_charts()
    if args.stage2:
        stage2_charts()


if __name__ == "__main__":
    main()
