import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import BM25_DIR
from retrievers import BM25Retriever, RescoringRetriever, rrf_fuse
from schema import Hit, Retriever, Scorer


class MockRetriever(Retriever):
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids

    def retrieve(self, query: str, k: int = 5) -> list[Hit]:
        return [Hit(c, 1.0 / r, r) for r, c in enumerate(self.ids[:k], start=1)]


class MockScorer(Scorer):
    def __init__(self, values: dict[str, float]) -> None:
        self.values = values

    def score(self, query: str, chunk_ids: list[str]) -> list[float]:
        return [self.values[c] for c in chunk_ids]


def main() -> int:
    failures = []

    fused = rrf_fuse([["A", "B", "C"], ["B", "C", "A"]], k=60, top_n=3)
    ids = [c for c, _ in fused]
    scores = [round(s, 5) for _, s in fused]
    if ids != ["B", "A", "C"] or scores != [0.03252, 0.03227, 0.03200]:
        failures.append(f"rrf_fuse wrong: {fused}")

    first = MockRetriever(["a", "b", "c", "d"])
    bm25 = MockRetriever(["d", "e", "a"])
    scorer = MockScorer({"a": 0.1, "b": 0.9, "c": 0.5, "d": 0.2, "e": 0.8})
    dense = RescoringRetriever("t", first, scorer, None, depth=4, rrf_k=60).retrieve("q", 3)
    if [h.chunk_id for h in dense] != ["b", "c", "d"] or [h.rank for h in dense] != [1, 2, 3]:
        failures.append(f"dense-only wrong: {dense}")
    hybrid = RescoringRetriever("t", first, scorer, bm25, depth=4, rrf_k=60).retrieve("q", 3)
    if [h.chunk_id for h in hybrid] != ["e", "d", "a"]:
        failures.append(f"hybrid ids wrong: {hybrid}")
    elif [round(h.score, 5) for h in hybrid] != [0.03226, 0.03202, 0.03126]:
        failures.append(f"hybrid scores wrong: {[h.score for h in hybrid]}")

    if (BM25_DIR / "fixed.pkl").exists():
        hits = BM25Retriever("fixed").retrieve("request body", 5)
        if len(hits) != 5 or [h.rank for h in hits] != [1, 2, 3, 4, 5]:
            failures.append(f"bm25 wrong: {hits}")
        else:
            print("bm25 on fixed: 5 hits")
    else:
        print("SKIP bm25 on fixed: artifacts/bm25/fixed.pkl not built yet")

    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
