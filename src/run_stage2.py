import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import time
from typing import Any

from generator import generate
from judge import judge_answerable, judge_refusal
from oracle_retriever import OracleRetriever
from prompts import format_context

try:
    from config import (
        RESULTS_DIR,
        TOP_K,
        GEN_MODEL,
        JUDGE_MODEL,
        SEED,
        CANDIDATE_DEPTH,
        RRF_K,
        TOP_CONFIGS_FOR_STAGE2,
        TOKENIZER_NAME,
        MAX_CHUNK_TOKENS,
        FIXED_OVERLAP_TOKENS,
        RECURSIVE_OVERLAP_TOKENS,
        SENTENCES_PER_CHUNK,
        SEMANTIC_PERCENTILE,
        SEMANTIC_SPLIT_MODEL,
        EMBEDDER_MAX_SEQ,
        GEN_NUM_CTX,
        GEN_MAX_TOKENS,
        JUDGE_MAX_TOKENS,
        parse_retrieval,
    )
except ImportError:
    ROOT = Path(__file__).resolve().parent.parent
    RESULTS_DIR = ROOT / "results"
    TOP_K = 5
    GEN_MODEL = os.environ.get("GEN_MODEL", "llama3.2:3b")
    JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "")
    SEED = 42
    CANDIDATE_DEPTH = 20
    RRF_K = 60
    TOP_CONFIGS_FOR_STAGE2 = 3
    TOKENIZER_NAME = "bert-base-uncased"
    MAX_CHUNK_TOKENS = 400
    FIXED_OVERLAP_TOKENS = 50
    RECURSIVE_OVERLAP_TOKENS = 50
    SENTENCES_PER_CHUNK = 5
    SEMANTIC_PERCENTILE = 20
    SEMANTIC_SPLIT_MODEL = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDER_MAX_SEQ = 512
    GEN_NUM_CTX = 8192
    GEN_MAX_TOKENS = 400
    JUDGE_MAX_TOKENS = 300

    def parse_retrieval(retrieval: str) -> tuple[str, bool]:
        return retrieval.split("_")[0], retrieval.endswith("_bm25")

try:
    from data_access import load_testset, chunk_index
except ImportError:
    ROOT = Path(__file__).resolve().parent.parent

    def load_testset() -> list[dict[str, Any]]:
        path = ROOT / "data" / "testset.jsonl"
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def chunk_index(chunker: str) -> dict[str, dict[str, Any]]:
        path = ROOT / "artifacts" / "chunks" / f"{chunker}.jsonl"
        if not path.is_file():
            return {}
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

def _load_completed_keys(runs_path: Path) -> set[tuple[str, str]]:
    completed: set[tuple[str, str]] = set()
    if not runs_path.is_file():
        return completed
    with open(runs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            completed.add((data["run_id"], data["question_id"]))
    return completed

def _write_metadata() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = RESULTS_DIR / "stage2_meta.json"
    meta = {
        "generator_model": GEN_MODEL,
        "judge_model": JUDGE_MODEL if JUDGE_MODEL else "unconfigured",
        "date": datetime.date.today().isoformat(),
        "constants": {
            "seed": SEED,
            "top_k": TOP_K,
            "candidate_depth": CANDIDATE_DEPTH,
            "rrf_k": RRF_K,
            "top_configs_for_stage2": TOP_CONFIGS_FOR_STAGE2,
            "tokenizer_name": TOKENIZER_NAME,
            "max_chunk_tokens": MAX_CHUNK_TOKENS,
            "fixed_overlap_tokens": FIXED_OVERLAP_TOKENS,
            "recursive_overlap_tokens": RECURSIVE_OVERLAP_TOKENS,
            "sentences_per_chunk": SENTENCES_PER_CHUNK,
            "semantic_percentile": SEMANTIC_PERCENTILE,
            "semantic_split_model": SEMANTIC_SPLIT_MODEL,
            "embedder_max_seq": EMBEDDER_MAX_SEQ,
            "gen_num_ctx": GEN_NUM_CTX,
            "gen_max_tokens": GEN_MAX_TOKENS,
            "judge_max_tokens": JUDGE_MAX_TOKENS,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def _aggregate_and_write_csv(runs_path: Path, csv_path: Path) -> int:
    if not runs_path.is_file():
        return 0
    records: list[dict[str, Any]] = []
    with open(runs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    grouped: dict[str, list[dict[str, Any]]] = {}
    total_judge_errors = 0
    for r in records:
        grouped.setdefault(r["run_id"], []).append(r)
        if r.get("judge_error"):
            total_judge_errors += 1

    fieldnames = [
        "run_id",
        "config_id",
        "prompt_style",
        "n_answerable",
        "n_unanswerable",
        "groundedness",
        "correctness",
        "refusal_accuracy",
        "retrieval_latency_ms",
        "generation_latency_ms",
        "prompt_tokens",
        "completion_tokens",
    ]

    rows: list[dict[str, Any]] = []
    for run_id, run_records in grouped.items():
        config_id = run_records[0]["config_id"]
        prompt_style = run_records[0]["prompt_style"]

        ans_records = [r for r in run_records if r.get("answerable")]
        unans_records = [r for r in run_records if not r.get("answerable")]

        n_ans = len(ans_records)
        n_unans = len(unans_records)

        grounded_scores: list[float] = []
        correct_scores: list[float] = []
        for r in ans_records:
            j = r.get("judge")
            if isinstance(j, dict):
                g = j.get("groundedness")
                c = j.get("correctness")
                if isinstance(g, (int, float)):
                    grounded_scores.append(float(g))
                if isinstance(c, (int, float)):
                    correct_scores.append(float(c))

        groundedness = (sum(grounded_scores) / len(grounded_scores)) if grounded_scores else 0.0
        correctness = (sum(correct_scores) / len(correct_scores)) if correct_scores else 0.0

        refusal_correct = 0
        for r in unans_records:
            j = r.get("judge")
            if isinstance(j, dict) and j.get("refused") is True:
                refusal_correct += 1
        refusal_accuracy = (refusal_correct / n_unans) if n_unans > 0 else 0.0

        retrieval_latency_ms = sum(r.get("retrieval_ms", 0.0) for r in run_records) / len(run_records)
        generation_latency_ms = sum(r.get("generation_ms", 0.0) for r in run_records) / len(run_records)
        prompt_tokens = sum(r.get("prompt_tokens", 0) for r in run_records) / len(run_records)
        completion_tokens = sum(r.get("completion_tokens", 0) for r in run_records) / len(run_records)

        rows.append({
            "run_id": run_id,
            "config_id": config_id,
            "prompt_style": prompt_style,
            "n_answerable": n_ans,
            "n_unanswerable": n_unans,
            "groundedness": f"{groundedness:.4f}",
            "correctness": f"{correctness:.4f}",
            "refusal_accuracy": f"{refusal_accuracy:.4f}",
            "retrieval_latency_ms": f"{retrieval_latency_ms:.2f}",
            "generation_latency_ms": f"{generation_latency_ms:.2f}",
            "prompt_tokens": f"{prompt_tokens:.1f}",
            "completion_tokens": f"{completion_tokens:.1f}",
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return total_judge_errors

def run_stage2(
    configs: list[str],
    styles: list[str],
    limit: int | None = None,
    resume: bool = False,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runs_path = RESULTS_DIR / "stage2_runs.jsonl"
    csv_path = RESULTS_DIR / "stage2.csv"

    completed_keys = _load_completed_keys(runs_path) if resume else set()
    questions = load_testset()
    if limit is not None and limit > 0:
        questions = questions[:limit]

    oracle_retriever: OracleRetriever | None = None
    chunk_caches: dict[str, dict[str, dict[str, Any]]] = {}

    for cfg in configs:
        if cfg == "oracle":
            if oracle_retriever is None:
                oracle_retriever = OracleRetriever()
            retriever = oracle_retriever
            chunker_name = "oracle"
        else:
            parts = cfg.split("__")
            if len(parts) != 3:
                raise ValueError(f"Invalid config string format: {cfg}")
            chunker_name, embedder_name, retrieval_name = parts
            if get_retriever is None:
                raise RuntimeError("get_retriever is not available in retrievers module")
            retriever = get_retriever(chunker_name, embedder_name, retrieval_name)

        if chunker_name not in chunk_caches and chunker_name != "oracle":
            chunk_caches[chunker_name] = chunk_index(chunker_name)

        for q in questions:
            qid = q["id"]
            q_text = q["question"]
            is_ans = bool(q.get("answerable", True))

            needs_run = False
            for style in styles:
                run_id = f"{cfg}__{style}"
                if (run_id, qid) not in completed_keys:
                    needs_run = True
                    break

            if not needs_run:
                continue

            t0 = time.perf_counter()
            hits = retriever.retrieve(q_text, k=TOP_K)
            retrieval_ms = (time.perf_counter() - t0) * 1000.0

            chunks: list[dict[str, Any]] = []
            for h in hits:
                if chunker_name == "oracle":
                    chunks.append(oracle_retriever.get_chunk_record(h.chunk_id))
                else:
                    rec = chunk_caches[chunker_name].get(h.chunk_id)
                    if rec is not None:
                        chunks.append(rec)
                    else:
                        chunks.append({"chunk_id": h.chunk_id, "text": ""})

            for style in styles:
                run_id = f"{cfg}__{style}"
                if (run_id, qid) in completed_keys:
                    continue

                gen_res = generate(style, q_text, chunks)
                answer = gen_res["answer"]
                ctx_text = format_context(chunks)

                if is_ans:
                    ref_ans = q.get("reference_answer", "")
                    judge_res = judge_answerable(q_text, ref_ans, ctx_text, answer)
                else:
                    judge_res = judge_refusal(q_text, answer)

                judge_err = judge_res.get("judge_error")
                if judge_err:
                    judge_obj = None
                else:
                    judge_obj = {k: v for k, v in judge_res.items() if k != "judge_error"}

                record = {
                    "run_id": run_id,
                    "config_id": cfg,
                    "prompt_style": style,
                    "question_id": qid,
                    "answerable": is_ans,
                    "answer": answer,
                    "retrieved_chunk_ids": [h.chunk_id for h in hits],
                    "retrieval_ms": float(retrieval_ms),
                    "generation_ms": float(gen_res["generation_ms"]),
                    "prompt_tokens": int(gen_res["prompt_tokens"]),
                    "completion_tokens": int(gen_res["completion_tokens"]),
                    "judge": judge_obj,
                    "judge_error": judge_err,
                }

                with open(runs_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
                completed_keys.add((run_id, qid))

    errors = _aggregate_and_write_csv(runs_path, csv_path)
    _write_metadata()
    print(f"Stage 2 run completed. Judge errors: {errors}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default=None)
    parser.add_argument("--styles", default="plain,grounded")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    configs_list: list[str] = []
    if args.configs is not None:
        configs_list = [c.strip() for c in args.configs.split(",") if c.strip()]
    else:
        top3_path = RESULTS_DIR / "top3.json"
        if top3_path.is_file():
            with open(top3_path, "r", encoding="utf-8") as f:
                top3_data = json.load(f)
            configs_list = [item["config_id"] for item in top3_data]
        else:
            configs_list = ["oracle"]

    styles_list = [s.strip() for s in args.styles.split(",") if s.strip()]
    run_stage2(configs_list, styles_list, limit=args.limit, resume=args.resume)

if __name__ == "__main__":
    main()
