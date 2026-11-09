import argparse
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import (CHUNKERS, EMBEDDERS, RESULTS_DIR, RETRIEVALS, SEED, TOP_CONFIGS_FOR_STAGE2,
                    TOP_K, config_id)
from analysis import stage1_table
from data_access import chunk_index, load_passages, load_testset
from metrics import gold_locations, score_retrieval
from retrievers import get_retriever

SUMMARY_COLS = ["config_id", "chunker", "embedder", "retrieval", "n_questions", "n_chunks",
                "context_recall", "context_precision", "hit_rate", "retrieval_latency_ms"]
PER_Q_COLS = ["config_id", "question_id", "context_recall", "context_precision", "hit",
              "latency_ms", "retrieved_chunk_ids"]


def all_config_ids() -> list[str]:
    return [config_id(c, e, r) for c in CHUNKERS for e in EMBEDDERS for r in RETRIEVALS]


def parse_config(cid: str) -> tuple[str, str, str]:
    chunker, embedder, retrieval = cid.split("__")
    return chunker, embedder, retrieval


def merge_csv(path: Path, new: pd.DataFrame, cols: list[str], replaced: set[str]) -> pd.DataFrame:
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old["config_id"].isin(replaced)]
        new = pd.concat([old, new], ignore_index=True)
    order = {cid: i for i, cid in enumerate(all_config_ids())}
    new = new.assign(_o=new["config_id"].map(lambda c: order.get(c, 10**6)))
    new = new.sort_values(["_o"], kind="stable").drop(columns="_o")
    new[cols].to_csv(path, index=False, encoding="utf-8")
    return new[cols]


def select_top(summary: pd.DataFrame, n: int = TOP_CONFIGS_FOR_STAGE2) -> pd.DataFrame:
    ranked = summary.sort_values(["context_recall", "context_precision", "retrieval_latency_ms"],
                                 ascending=[False, False, True], kind="stable")
    return ranked.head(n)


def run_config(cid: str, questions: list[dict], locs: dict[str, list], k: int,
               cindex_cache: dict[str, dict]) -> tuple[list[dict], dict]:
    chunker, embedder, retrieval = parse_config(cid)
    if chunker not in cindex_cache:
        cindex_cache.clear()
        cindex_cache[chunker] = chunk_index(chunker)
    cindex = cindex_cache[chunker]
    retriever = get_retriever(chunker, embedder, retrieval)
    retriever.retrieve(questions[0]["question"], k)
    rows = []
    for q in questions:
        t0 = time.perf_counter()
        hits = retriever.retrieve(q["question"], k)
        latency = (time.perf_counter() - t0) * 1000.0
        chunks = [cindex[h.chunk_id] for h in hits]
        s = score_retrieval(chunks, locs[q["id"]])
        rows.append({
            "config_id": cid, "question_id": q["id"],
            "context_recall": s["context_recall"], "context_precision": s["context_precision"],
            "hit": int(s["hit"]), "latency_ms": latency,
            "retrieved_chunk_ids": json.dumps([h.chunk_id for h in hits]),
        })
    del retriever
    gc.collect()
    df = pd.DataFrame(rows)
    summary = {
        "config_id": cid, "chunker": chunker, "embedder": embedder, "retrieval": retrieval,
        "n_questions": len(df), "n_chunks": len(cindex),
        "context_recall": df["context_recall"].mean(),
        "context_precision": df["context_precision"].mean(),
        "hit_rate": df["hit"].mean(),
        "retrieval_latency_ms": df["latency_ms"].mean(),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--k", type=int, default=TOP_K)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    configs = all_config_ids() if args.configs == "all" else [c.strip() for c in args.configs.split(",")]
    unknown = [c for c in configs if c not in set(all_config_ids())]
    if unknown:
        raise SystemExit(f"unknown config ids: {unknown}")

    questions = [q for q in load_testset() if q["answerable"]]
    if args.limit:
        questions = questions[: args.limit]
    passages = {}
    locs = {}
    for q in questions:
        g = q["gold_evidence"]
        if g["source"] not in passages:
            passages[g["source"]] = load_passages(g["source"])
        locs[q["id"]] = gold_locations(passages[g["source"]], g)
        if not locs[q["id"]]:
            raise SystemExit(f"no gold location for {q['id']}")
    print(f"questions: {len(questions)}, configurations: {len(configs)}")

    per_q_rows: list[dict] = []
    summaries: list[dict] = []
    cindex_cache: dict[str, dict] = {}
    for i, cid in enumerate(configs, start=1):
        t0 = time.perf_counter()
        rows, summary = run_config(cid, questions, locs, args.k, cindex_cache)
        per_q_rows.extend(rows)
        summaries.append(summary)
        print(f"[{i}/{len(configs)}] {cid}: recall {summary['context_recall']:.3f} "
              f"precision {summary['context_precision']:.3f} "
              f"latency {summary['retrieval_latency_ms']:.1f} ms ({time.perf_counter() - t0:.1f}s)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    replaced = set(configs)
    merge_csv(RESULTS_DIR / "stage1_per_question.csv", pd.DataFrame(per_q_rows), PER_Q_COLS, replaced)
    summary_df = merge_csv(RESULTS_DIR / "stage1.csv", pd.DataFrame(summaries), SUMMARY_COLS, replaced)

    stage1_table()
    ranked = summary_df.sort_values(["context_recall", "context_precision", "retrieval_latency_ms"],
                                    ascending=[False, False, True], kind="stable")
    print(ranked[["config_id", "context_recall", "context_precision", "hit_rate",
                  "retrieval_latency_ms"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    top = select_top(summary_df)
    top3 = [{"config_id": r.config_id, "chunker": r.chunker, "embedder": r.embedder,
             "retrieval": r.retrieval} for r in top.itertuples()]
    with open(RESULTS_DIR / "top3.json", "w", encoding="utf-8") as f:
        json.dump(top3, f, indent=2)
    if len(summary_df) < len(all_config_ids()):
        print(f"WARNING: {len(summary_df)} of {len(all_config_ids())} configurations present, "
              "selection is provisional")
    print(f"done: wrote stage1.csv ({len(summary_df)} rows), stage1_per_question.csv, stage1_table.md, top3.json: "
          f"{[t['config_id'] for t in top3]}")


if __name__ == "__main__":
    main()
