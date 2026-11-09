import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_access import load_passages, load_testset
from metrics import gold_locations, score_retrieval


def main() -> int:
    failures = []
    passages = {s: load_passages(s) for s in ("fastapi", "pytorch")}
    answerable = [q for q in load_testset() if q["answerable"]]
    locs_by_q = {}
    for q in answerable:
        g = q["gold_evidence"]
        locs = gold_locations(passages[g["source"]], g)
        locs_by_q[q["id"]] = locs
        if not locs:
            failures.append(f"no gold location for {q['id']}")
    print(f"answerable questions: {len(answerable)}")

    q = answerable[0]
    pidx, ga, gb = locs_by_q[q["id"]][0]
    plen = len(passages[q["gold_evidence"]["source"]][pidx]["text"])

    cover = {"passage_idx": pidx, "char_start": max(0, ga - 10), "char_end": min(plen, gb + 10)}
    r = score_retrieval([cover], locs_by_q[q["id"]])
    if not (r["context_recall"] == 1.0 and r["context_precision"] == 1.0 and r["hit"]):
        failures.append(f"full cover failed: {r}")

    mid = ga + (gb - ga) // 2
    half = {"passage_idx": pidx, "char_start": ga, "char_end": mid}
    other = {"passage_idx": -1, "char_start": 0, "char_end": 50}
    r = score_retrieval([half, other], locs_by_q[q["id"]])
    expected = (mid - ga) / (gb - ga)
    if not (abs(r["context_recall"] - expected) < 1e-9 and abs(expected - 0.5) < 0.05
            and r["context_precision"] == 0.5 and r["hit"]):
        failures.append(f"half cover failed: {r}")

    r = score_retrieval([other], locs_by_q[q["id"]])
    if not (r["context_recall"] == 0.0 and r["context_precision"] == 0.0 and not r["hit"]):
        failures.append(f"unrelated failed: {r}")

    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
