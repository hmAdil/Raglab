import argparse
import csv
import json
from pathlib import Path
import random
from typing import Any

from oracle_retriever import OracleRetriever
from prompts import format_context

try:
    from config import RESULTS_DIR
except ImportError:
    ROOT = Path(__file__).resolve().parent.parent
    RESULTS_DIR = ROOT / "results"

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

def make_validation_sheet() -> None:
    runs_path = RESULTS_DIR / "stage2_runs.jsonl"
    sheet_path = RESULTS_DIR / "judge_validation_sheet.csv"
    if not runs_path.is_file():
        print(f"Cannot create sheet: {runs_path} does not exist.")
        return

    eligible: list[dict[str, Any]] = []
    with open(runs_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get("answerable"):
                continue
            if rec.get("judge_error") is not None:
                continue
            j = rec.get("judge")
            if not isinstance(j, dict):
                continue
            g = j.get("groundedness")
            c = j.get("correctness")
            if isinstance(g, bool) or not isinstance(g, int) or g < 1 or g > 5:
                continue
            if isinstance(c, bool) or not isinstance(c, int) or c < 1 or c > 5:
                continue
            eligible.append(rec)

    if len(eligible) < 10:
        print(f"Found {len(eligible)} eligible records. At least 10 required to create validation sheet.")
        return

    rng = random.Random(42)
    selected = rng.sample(eligible, 10)

    testset = load_testset()
    q_lookup = {q["id"]: q for q in testset}

    oracle_retriever: OracleRetriever | None = None
    chunk_caches: dict[str, dict[str, dict[str, Any]]] = {}

    rows: list[dict[str, Any]] = []
    for idx, rec in enumerate(selected, start=1):
        run_id = rec["run_id"]
        qid = rec["question_id"]
        q_item = q_lookup.get(qid, {})
        question_text = q_item.get("question", "")
        ref_ans = q_item.get("reference_answer", "")
        gen_ans = rec.get("answer", "")
        cfg_id = rec["config_id"]
        chunk_ids = rec.get("retrieved_chunk_ids", [])

        chunk_records: list[dict[str, Any]] = []
        if cfg_id == "oracle" or any(cid.startswith("oracle-") for cid in chunk_ids):
            if oracle_retriever is None:
                oracle_retriever = OracleRetriever()
            for cid in chunk_ids:
                chunk_records.append(oracle_retriever.get_chunk_record(cid))
        else:
            chunker_name = cfg_id.split("__")[0]
            if chunker_name not in chunk_caches:
                chunk_caches[chunker_name] = chunk_index(chunker_name)
            lookup = chunk_caches[chunker_name]
            for cid in chunk_ids:
                if cid not in lookup:
                    raise RuntimeError(f"Missing chunk artifact for {cid} in chunker {chunker_name}")
                chunk_records.append(lookup[cid])

        ctx_str = format_context(chunk_records)
        rows.append({
            "row_id": idx,
            "run_id": run_id,
            "question_id": qid,
            "question": question_text,
            "reference_answer": ref_ans,
            "retrieved_context": ctx_str,
            "generated_answer": gen_ans,
            "manual_groundedness": "",
            "manual_correctness": "",
        })

    fieldnames = [
        "row_id",
        "run_id",
        "question_id",
        "question",
        "reference_answer",
        "retrieved_context",
        "generated_answer",
        "manual_groundedness",
        "manual_correctness",
    ]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(sheet_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Created validation sheet with 10 samples: {sheet_path}")

def generate_validation_report() -> None:
    sheet_path = RESULTS_DIR / "judge_validation_sheet.csv"
    runs_path = RESULTS_DIR / "stage2_runs.jsonl"
    out_json_path = RESULTS_DIR / "judge_validation.json"

    if not sheet_path.is_file():
        print(f"Cannot generate report: {sheet_path} does not exist.")
        return
    if not runs_path.is_file():
        print(f"Cannot generate report: {runs_path} does not exist.")
        return

    runs_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    with open(runs_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                runs_lookup[(rec["run_id"], rec["question_id"])] = rec

    sheet_rows: list[dict[str, Any]] = []
    with open(sheet_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            sheet_rows.append(r)

    if not sheet_rows:
        print("Validation sheet is empty.")
        return

    manual_g: list[int] = []
    manual_c: list[int] = []
    judge_g: list[int] = []
    judge_c: list[int] = []

    for r in sheet_rows:
        key = (r["run_id"], r["question_id"])
        if key not in runs_lookup:
            raise KeyError(f"Run key {key} not found in Stage 2 runs")
        orig = runs_lookup[key]
        j_obj = orig.get("judge")
        if not isinstance(j_obj, dict):
            raise ValueError(f"Judge output missing for run {key}")

        raw_mg = r["manual_groundedness"].strip()
        raw_mc = r["manual_correctness"].strip()
        if not raw_mg or not raw_mc:
            raise ValueError(f"Incomplete manual score at row {r.get('row_id')}")

        mg = int(raw_mg)
        mc = int(raw_mc)
        if mg < 1 or mg > 5 or mc < 1 or mc > 5:
            raise ValueError(f"Manual scores must be integers between 1 and 5 at row {r.get('row_id')}")

        jg = int(j_obj["groundedness"])
        jc = int(j_obj["correctness"])

        manual_g.append(mg)
        manual_c.append(mc)
        judge_g.append(jg)
        judge_c.append(jc)

    n_samples = len(sheet_rows)

    g_exact = (sum(1 for m, j in zip(manual_g, judge_g) if m == j) / n_samples) * 100.0
    g_within_one = (sum(1 for m, j in zip(manual_g, judge_g) if abs(m - j) <= 1) / n_samples) * 100.0
    g_mae = sum(abs(m - j) for m, j in zip(manual_g, judge_g)) / n_samples

    c_exact = (sum(1 for m, j in zip(manual_c, judge_c) if m == j) / n_samples) * 100.0
    c_within_one = (sum(1 for m, j in zip(manual_c, judge_c) if abs(m - j) <= 1) / n_samples) * 100.0
    c_mae = sum(abs(m - j) for m, j in zip(manual_c, judge_c)) / n_samples

    report_data = {
        "n_samples": n_samples,
        "groundedness": {
            "exact_agreement": round(g_exact, 2),
            "within_one_agreement": round(g_within_one, 2),
            "mean_abs_diff": round(g_mae, 4),
        },
        "correctness": {
            "exact_agreement": round(c_exact, 2),
            "within_one_agreement": round(c_within_one, 2),
            "mean_abs_diff": round(c_mae, 4),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"Judge Validation Report (N={n_samples}):")
    print(f"  Groundedness: Exact={g_exact:.1f}%, Within-1={g_within_one:.1f}%, MAE={g_mae:.3f}")
    print(f"  Correctness:  Exact={c_exact:.1f}%, Within-1={c_within_one:.1f}%, MAE={c_mae:.3f}")
    print("\nDiscussion of Known LLM Judge Biases:")
    print("1. Verbosity Bias: LLM judges systematically favor longer, verbose responses over concise ones.")
    print("2. Position Bias: Candidate order can alter comparative judgments; pointwise 1-5 scoring mitigates this.")
    print("3. Self-Enhancement Bias: Models tend to score outputs generated by their own family higher.")
    print("4. Pointwise vs Pairwise: RAGAS validates faithfulness on pairwise binary comparisons; RAGLab evaluates absolute pointwise 1-5 scales.")
    print("5. Manual Validation Necessity: Empirically grounds automated metric trustworthiness against human rubrics.")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-sheet", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    if args.make_sheet:
        make_validation_sheet()
    elif args.report:
        generate_validation_report()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
