import argparse

import pandas as pd

from config import RESULTS_DIR


def load_stage1() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "stage1.csv")


def load_stage2() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "stage2.csv")


def minmax(s: pd.Series) -> pd.Series:
    span = s.max() - s.min()
    if span == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / span


def decision_table() -> pd.DataFrame:
    df = load_stage2().copy()
    df["total_latency_ms"] = df["retrieval_latency_ms"] + df["generation_latency_ms"]
    acc = df.sort_values(["correctness", "groundedness"], ascending=False, kind="stable").iloc[0]
    fast = df.sort_values("total_latency_ms", ascending=True, kind="stable").iloc[0]
    df["balance"] = minmax(df["correctness"]) - minmax(df["total_latency_ms"])
    bal = df.sort_values("balance", ascending=False, kind="stable").iloc[0]
    rows = [
        ("best_accuracy", acc, "highest correctness, ties broken by groundedness"),
        ("best_speed", fast, "lowest retrieval plus generation latency"),
        ("best_balance", bal, "max of min-max correctness minus min-max total latency over the runs"),
    ]
    out = pd.DataFrame([{
        "criterion": name, "run_id": r["run_id"], "groundedness": r["groundedness"],
        "correctness": r["correctness"], "total_latency_ms": r["total_latency_ms"], "note": note,
    } for name, r, note in rows])
    out.to_csv(RESULTS_DIR / "decision_table.csv", index=False, encoding="utf-8")
    return out


def stage1_table() -> str:
    df = load_stage1().round(3)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    text = "\n".join(lines) + "\n"
    with open(RESULTS_DIR / "stage1_table.md", "w", encoding="utf-8") as f:
        f.write(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision", action="store_true")
    parser.add_argument("--stage1-table", action="store_true")
    args = parser.parse_args()
    if not (args.decision or args.stage1_table):
        parser.error("pass --decision and/or --stage1-table")
    if args.stage1_table:
        stage1_table()
        print("wrote results/stage1_table.md")
    if args.decision:
        out = decision_table()
        print(out.to_string(index=False))
        print("wrote results/decision_table.csv")


if __name__ == "__main__":
    main()
