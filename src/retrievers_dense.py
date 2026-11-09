import argparse
import json
from pathlib import Path

import numpy as np

from config import EMBEDDERS, INDEX_DIR, RESCORING_DIR, TOP_K
from embedders import ColBertEmbedder, SentenceEmbedder
from schema import Hit, Retriever, Scorer


def read_ids(directory: Path) -> list[str]:
    with (directory / "chunk_ids.json").open(encoding="utf-8") as handle:
        ids = json.load(handle)
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate chunk IDs: {directory}")
    return ids


def maxsim_scores(q: np.ndarray, tokens: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    sim = q @ tokens.T
    per_chunk_max = np.maximum.reduceat(sim, offsets[:-1], axis=1)
    return per_chunk_max.sum(axis=0)


def maxsim_subset(q: np.ndarray, tokens: np.ndarray, offsets: np.ndarray, rows: list[int]) -> list[float]:
    out = []
    for i in rows:
        d = tokens[offsets[i]:offsets[i + 1]]
        out.append(float((q @ d.T).max(axis=1).sum()))
    return out


class FirstStageRetriever(Retriever):
    def __init__(self, chunker: str, embedder: str) -> None:
        import faiss

        if embedder not in EMBEDDERS:
            raise ValueError(f"Not a first-stage embedder: {embedder}")
        directory = INDEX_DIR / f"{chunker}__{embedder}"
        self.chunk_ids = read_ids(directory)
        self.index = faiss.read_index(str(directory / "index.faiss"))
        if self.index.ntotal != len(self.chunk_ids):
            raise ValueError("FAISS rows and chunk IDs disagree")
        self.encoder = SentenceEmbedder(embedder)
        self.encoder.model
        self.config_id = f"{chunker}__{embedder}__first_stage"

    def retrieve(self, query: str, k: int = TOP_K) -> list[Hit]:
        if k < 1:
            raise ValueError("k must be positive")
        vector = self.encoder.encode_queries([query])
        scores, rows = self.index.search(np.ascontiguousarray(vector), min(k, len(self.chunk_ids)))
        return [Hit(self.chunk_ids[int(row)], float(score), rank) for rank, (row, score) in enumerate(zip(rows[0], scores[0]), start=1) if row >= 0]


class DistilBertScorer(Scorer):
    def __init__(self, chunker: str) -> None:
        directory = RESCORING_DIR / f"{chunker}__distilbert"
        self.chunk_ids = read_ids(directory)
        self.rows = {cid: i for i, cid in enumerate(self.chunk_ids)}
        self.vectors = np.load(directory / "vectors.npy")
        if self.vectors.shape != (len(self.chunk_ids), 768):
            raise ValueError("DistilBERT vectors and chunk IDs disagree")
        self.encoder = SentenceEmbedder("distilbert")
        self.encoder.model
        self.scorer_id = f"{chunker}__distilbert"

    def score(self, query: str, chunk_ids: list[str]) -> list[float]:
        if not chunk_ids:
            return []
        rows = [self.rows[cid] for cid in chunk_ids]
        vector = self.encoder.encode_queries([query])[0]
        return [float(value) for value in self.vectors[rows] @ vector]


class ColBertScorer(Scorer):
    def __init__(self, chunker: str) -> None:
        directory = RESCORING_DIR / f"{chunker}__colbert"
        self.chunk_ids = read_ids(directory)
        self.rows = {cid: i for i, cid in enumerate(self.chunk_ids)}
        self.tokens = np.asarray(np.load(directory / "tokens.npy"), dtype=np.float32)
        self.offsets = np.load(directory / "offsets.npy")
        if self.tokens.ndim != 2 or self.tokens.shape[1] != 128 or self.offsets.shape != (len(self.chunk_ids) + 1,) or self.offsets[0] != 0 or self.offsets[-1] != len(self.tokens) or not np.all(np.diff(self.offsets) > 0):
            raise ValueError("Invalid ColBERT store")
        self.encoder = ColBertEmbedder()
        self.encoder.model
        self.scorer_id = f"{chunker}__colbert"

    def score(self, query: str, chunk_ids: list[str]) -> list[float]:
        if not chunk_ids:
            return []
        rows = [self.rows[cid] for cid in chunk_ids]
        return maxsim_subset(self.encoder.encode_query(query), self.tokens, self.offsets, rows)


def get_first_stage(chunker: str, embedder: str) -> Retriever:
    return FirstStageRetriever(chunker, embedder)


def get_scorer(chunker: str, rescorer: str) -> Scorer:
    if rescorer == "distilbert":
        return DistilBertScorer(chunker)
    if rescorer == "colbert":
        return ColBertScorer(chunker)
    raise ValueError(f"Unknown rescorer: {rescorer}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunker", default="fixed")
    parser.add_argument("--embedder", choices=EMBEDDERS, default="bge")
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=TOP_K)
    args = parser.parse_args()
    for hit in get_first_stage(args.chunker, args.embedder).retrieve(args.query, args.k):
        print(json.dumps({"chunk_id": hit.chunk_id, "rank": hit.rank, "score": hit.score}))


if __name__ == "__main__":
    main()
