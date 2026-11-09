import argparse
import pickle
import re

from rank_bm25 import BM25Okapi

from config import BM25_DIR, CANDIDATE_DEPTH, CHUNKERS, RRF_K, TOP_K, config_id, parse_retrieval
from data_access import load_chunks
from schema import Hit, Retriever, Scorer


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def rrf_fuse(rankings: list[list[str]], k: int = 60, top_n: int = 5) -> list[tuple[str, float]]:
    scores = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return ordered[:top_n]


def build_bm25(chunker: str) -> int:
    chunks = load_chunks(chunker)
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    BM25_DIR.mkdir(parents=True, exist_ok=True)
    with open(BM25_DIR / f"{chunker}.pkl", "wb") as f:
        pickle.dump({"chunk_ids": [c["chunk_id"] for c in chunks], "bm25": bm25}, f)
    return len(chunks)


class BM25Retriever(Retriever):
    def __init__(self, chunker: str) -> None:
        with open(BM25_DIR / f"{chunker}.pkl", "rb") as f:
            data = pickle.load(f)
        self.chunk_ids: list[str] = data["chunk_ids"]
        self.bm25: BM25Okapi = data["bm25"]
        self.config_id = f"{chunker}__bm25"

    def retrieve(self, query: str, k: int = 5) -> list[Hit]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(self.chunk_ids)), key=lambda i: (-float(scores[i]), self.chunk_ids[i]))
        return [Hit(self.chunk_ids[i], float(scores[i]), r) for r, i in enumerate(order[:k], start=1)]


def rescoring_retrieve(query: str, first_stage: Retriever, bm25: Retriever | None, scorer: Scorer,
                       depth: int, k: int, rrf_k: int, use_bm25: bool) -> list[Hit]:
    pool = [h.chunk_id for h in first_stage.retrieve(query, depth)]
    bm25_ids: list[str] = []
    if use_bm25:
        assert bm25 is not None
        bm25_ids = [h.chunk_id for h in bm25.retrieve(query, depth)]
        seen = set(pool)
        for cid in bm25_ids:
            if cid not in seen:
                pool.append(cid)
                seen.add(cid)
    scores = scorer.score(query, pool)
    order = sorted(range(len(pool)), key=lambda i: (-scores[i], pool[i]))
    ranking = [pool[i] for i in order]
    if not use_bm25:
        return [Hit(pool[i], float(scores[i]), r) for r, i in enumerate(order[:k], start=1)]
    fused = rrf_fuse([ranking, bm25_ids], rrf_k, k)
    return [Hit(cid, float(s), r) for r, (cid, s) in enumerate(fused, start=1)]


class RescoringRetriever(Retriever):
    def __init__(self, cfg_id: str, first_stage: Retriever, scorer: Scorer, bm25: Retriever | None,
                 depth: int = CANDIDATE_DEPTH, rrf_k: int = RRF_K) -> None:
        self.config_id = cfg_id
        self.first_stage = first_stage
        self.scorer = scorer
        self.bm25 = bm25
        self.depth = depth
        self.rrf_k = rrf_k

    def retrieve(self, query: str, k: int = TOP_K) -> list[Hit]:
        return rescoring_retrieve(query, self.first_stage, self.bm25, self.scorer,
                                  self.depth, k, self.rrf_k, self.bm25 is not None)


def get_retriever(chunker: str, embedder: str, retrieval: str) -> Retriever:
    from retrievers_dense import get_first_stage, get_scorer

    rescorer, use_bm25 = parse_retrieval(retrieval)
    first_stage = get_first_stage(chunker, embedder)
    scorer = get_scorer(chunker, rescorer)
    bm25 = BM25Retriever(chunker) if use_bm25 else None
    return RescoringRetriever(config_id(chunker, embedder, retrieval), first_stage, scorer, bm25)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-bm25", default=None, help="chunker name or all")
    args = parser.parse_args()
    if args.build_bm25:
        names = CHUNKERS if args.build_bm25 == "all" else (args.build_bm25,)
        for name in names:
            n = build_bm25(name)
            print(f"bm25 {name}: {n} chunks")
        print(f"built bm25 for {len(names)} chunkers")


if __name__ == "__main__":
    main()
