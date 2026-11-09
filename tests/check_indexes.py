import argparse
import gc
import re
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import CANDIDATE_DEPTH, CHUNKERS, EMBEDDERS, INDEX_DIR, RESCORING_DIR, SEED, SOURCES, TOP_K
from data_access import chunk_index, load_chunks, load_passages, load_testset
from embedders import normalize_rows
from retrievers_dense import ColBertScorer, DistilBertScorer, get_first_stage, get_scorer, maxsim_scores, maxsim_subset, read_ids


def normalize_section(s: str) -> str:
    s = (s or "").replace("\ufffd", "").replace("\u00b6", "").replace("#", "")
    return re.sub(r"\s+", " ", s).strip().lower()


def find_all(text: str, span: str) -> list[tuple[int, int]]:
    out = []
    start = text.find(span)
    while start != -1:
        out.append((start, start + len(span)))
        start = text.find(span, start + 1)
    return out


def gold_locations(passages: list[dict], gold: dict) -> list[tuple[int, int, int]]:
    sec = normalize_section(gold["section"])
    span = gold["span"]
    locs = []
    for idx, passage in enumerate(passages):
        if normalize_section(passage["section"]) == sec:
            for a, b in find_all(passage["text"], span):
                locs.append((idx, a, b))
    if locs:
        return locs
    for idx, passage in enumerate(passages):
        for a, b in find_all(passage["text"], span):
            locs.append((idx, a, b))
    return locs


def score_retrieval(retrieved_chunks: list[dict], locs: list[tuple[int, int, int]]) -> dict:
    best_cov = 0.0
    hit = False
    for pidx, ga, gb in locs:
        covered = np.zeros(gb - ga, dtype=bool)
        for chunk in retrieved_chunks:
            if chunk["passage_idx"] != pidx:
                continue
            a = max(ga, chunk["char_start"])
            b = min(gb, chunk["char_end"])
            if b > a:
                covered[a - ga:b - ga] = True
        cov = covered.mean()
        best_cov = max(best_cov, cov)
        hit = hit or cov > 0
    n_rel = 0
    for chunk in retrieved_chunks:
        for pidx, ga, gb in locs:
            if chunk["passage_idx"] == pidx and min(gb, chunk["char_end"]) > max(ga, chunk["char_start"]):
                n_rel += 1
                break
    precision = n_rel / len(retrieved_chunks) if retrieved_chunks else 0.0
    return {"context_recall": float(best_cov), "hit": bool(hit), "context_precision": precision}


def check_norms(vectors: np.ndarray, tolerance: float = 0.002) -> None:
    assert vectors.ndim == 2 and np.isfinite(vectors).all()
    for start in range(0, len(vectors), 4096):
        assert np.allclose(np.linalg.norm(np.asarray(vectors[start:start + 4096], dtype=np.float32), axis=1), 1.0, atol=tolerance)


def check_stores(chunker: str) -> None:
    import faiss

    ids = [chunk["chunk_id"] for chunk in load_chunks(chunker)]
    for model in EMBEDDERS:
        directory = INDEX_DIR / f"{chunker}__{model}"
        assert read_ids(directory) == ids
        index = faiss.read_index(str(directory / "index.faiss"))
        assert isinstance(index, faiss.IndexFlatIP) and index.d == 384 and index.ntotal == len(ids)
        check_norms(index.reconstruct_n(0, index.ntotal))
        retriever = get_first_stage(chunker, model)
        for k in (TOP_K, CANDIDATE_DEPTH, 50):
            hits = retriever.retrieve("How do I declare a request body?", k)
            assert len(hits) == min(k, len(ids))
            assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))
            assert len({hit.chunk_id for hit in hits}) == len(hits)
            assert all(hit.chunk_id in ids and np.isfinite(hit.score) for hit in hits)
            assert all(a.score >= b.score for a, b in zip(hits, hits[1:]))
        del retriever
        gc.collect()
    for rescorer in ("distilbert", "colbert"):
        directory = RESCORING_DIR / f"{chunker}__{rescorer}"
        assert read_ids(directory) == ids
        if rescorer == "distilbert":
            vectors = np.load(directory / "vectors.npy", mmap_mode="r")
            assert vectors.shape == (len(ids), 768) and vectors.dtype == np.float32
        else:
            vectors = np.load(directory / "tokens.npy", mmap_mode="r")
            offsets = np.load(directory / "offsets.npy")
            assert vectors.ndim == 2 and vectors.shape[1] == 128 and vectors.dtype == np.float16
            assert offsets.dtype == np.int64 and offsets.shape == (len(ids) + 1,)
            assert offsets[0] == 0 and offsets[-1] == len(vectors) and np.all(np.diff(offsets) > 0)
        check_norms(vectors)
        del vectors
        scorer = get_scorer(chunker, rescorer)
        candidates = ids[:CANDIDATE_DEPTH]
        first = scorer.score("How do I declare a request body?", candidates)
        second = scorer.score("How do I declare a request body?", candidates)
        assert len(first) == len(candidates) and np.isfinite(first).all()
        assert np.allclose(first, second, atol=1e-5)
        assert scorer.score("empty pool", []) == []
        del scorer
        gc.collect()
    print(f"PASS {chunker}: row alignment, norms, ranks and scorer repeatability", flush=True)


def quality_checks() -> None:
    questions = [q for q in load_testset() if q["answerable"]]
    passages = {source: load_passages(source) for source in SOURCES}
    lookup = chunk_index("fixed")
    located = {q["id"]: gold_locations(passages[q["gold_evidence"]["source"]], q["gold_evidence"]) for q in questions}
    assert len(questions) == 27 and all(located.values())
    for embedder in EMBEDDERS:
        retriever = get_first_stage("fixed", embedder)
        pools = {q["id"]: [h.chunk_id for h in retriever.retrieve(q["question"], CANDIDATE_DEPTH)] for q in questions}
        del retriever
        gc.collect()
        def is_hit(question: dict, ids: list[str]) -> bool:
            chunks = [lookup[cid] for cid in ids if lookup[cid]["source"] == question["gold_evidence"]["source"]]
            return score_retrieval(chunks, located[question["id"]])["hit"]

        n_hits = sum(is_hit(q, pools[q["id"]]) for q in questions)
        print(f"{embedder} first-stage top {CANDIDATE_DEPTH}: {n_hits}/27", flush=True)
        assert n_hits >= 15, f"Low candidate recall for {embedder}: {n_hits}/27"
        for rescorer in ("distilbert", "colbert"):
            scorer = get_scorer("fixed", rescorer)
            successes = 0
            for question in questions:
                ids = pools[question["id"]]
                values = scorer.score(question["question"], ids)
                order = sorted(range(len(ids)), key=lambda i: (-values[i], ids[i]))
                successes += is_hit(question, [ids[i] for i in order[:TOP_K]])
            print(f"{embedder} + {rescorer} top {TOP_K}: {successes}/27", flush=True)
            assert successes >= 10, f"Low rescoring hit rate: {embedder}/{rescorer}: {successes}/27"
            del scorer
            gc.collect()


def unit_checks() -> None:
    rng = np.random.default_rng(SEED)
    q = normalize_rows(rng.normal(size=(7, 128)))
    tokens = normalize_rows(rng.normal(size=(23, 128)))
    offsets = np.asarray([0, 3, 12, 23], dtype=np.int64)
    expected = [float((q @ tokens[a:b].T).max(axis=1).sum()) for a, b in zip(offsets, offsets[1:])]
    assert np.allclose(maxsim_scores(q, tokens, offsets), expected, atol=1e-5)
    assert np.allclose(maxsim_subset(q, tokens, offsets, [2, 0, 2]), [expected[2], expected[0], expected[2]], atol=1e-5)
    assert maxsim_subset(q, tokens, offsets, []) == []
    class QueryEncoder:
        def __init__(self) -> None:
            self.calls = 0

        def encode_queries(self, texts: list[str]) -> np.ndarray:
            self.calls += 1
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

        def encode_query(self, text: str) -> np.ndarray:
            self.calls += 1
            return q

    distil = DistilBertScorer.__new__(DistilBertScorer)
    distil.rows = {"a": 0, "b": 1}
    distil.vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    distil.encoder = QueryEncoder()
    assert distil.score("query", ["b", "a", "b"]) == [0.0, 1.0, 0.0]
    assert distil.score("query", ["a"]) == [1.0] and distil.encoder.calls == 2
    colbert = ColBertScorer.__new__(ColBertScorer)
    colbert.rows = {"a": 0, "b": 1, "c": 2}
    colbert.tokens, colbert.offsets, colbert.encoder = tokens, offsets, QueryEncoder()
    assert np.allclose(colbert.score("query", ["c", "a"]), [expected[2], expected[0]])
    assert np.allclose(colbert.score("query", ["a"]), [expected[0]]) and colbert.encoder.calls == 2
    for scorer in (distil, colbert):
        assert scorer.score("query", []) == []
        try:
            scorer.score("query", ["missing"])
        except KeyError:
            pass
        else:
            raise AssertionError("Unknown chunk ID accepted")
    check_index_writes()
    print("PASS offline MaxSim, candidate order, duplicate IDs and query re-encoding checks", flush=True)


def check_index_writes() -> None:
    import indexer

    chunks = [{"chunk_id": f"fixed-{i:06d}", "text": str(i), "n_bert_tokens": n} for i, n in enumerate((40, 2, 20))]
    class SentenceEncoder:
        def __init__(self, model_id: str) -> None:
            pass

        def encode_documents(self, texts: list[str]) -> np.ndarray:
            result = np.zeros((len(texts), 768), dtype=np.float32)
            for i, text in enumerate(texts):
                result[i, int(text)] = 1.0
            return result

    class TokenEncoder:
        def encode_documents(self, texts: list[str]) -> list[np.ndarray]:
            output = []
            for text in texts:
                row = int(text)
                result = np.zeros((row + 1, 128), dtype=np.float32)
                result[:, row] = 1.0
                output.append(result)
            return output

    scratch = Path(__file__).resolve().parents[1] / "cache"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as temporary:
        root = Path(temporary)
        with patch.object(indexer, "RESCORING_DIR", root), patch.object(indexer, "load_chunks", return_value=chunks), patch.object(indexer, "SentenceEmbedder", SentenceEncoder), patch.object(indexer, "ColBertEmbedder", TokenEncoder):
            distil_path = indexer.build_index("fixed", "distilbert")
            assert np.array_equal(np.load(distil_path / "vectors.npy")[:, :3], np.eye(3))
            token_path = indexer.build_index("fixed", "colbert")
            assert np.array_equal(np.load(token_path / "offsets.npy"), [0, 1, 3, 6])
            values = np.load(token_path / "tokens.npy")
            assert np.array_equal(values.argmax(axis=1), [0, 1, 1, 2, 2, 2])
            expected_ids = [chunk["chunk_id"] for chunk in chunks]
            assert read_ids(distil_path) == read_ids(token_path) == expected_ids
            with patch.object(indexer, "ColBertEmbedder", side_effect=AssertionError("Complete artifacts must be skipped")):
                assert indexer.build_index("fixed", "colbert") == token_path
            (token_path / "offsets.npy").unlink()
            indexer.build_index("fixed", "colbert")
            assert indexer.complete_artifact(token_path, "colbert", expected_ids)
            assert not indexer.complete_artifact(token_path, "colbert", list(reversed(expected_ids)))
    print("PASS offline index writes: original row order, offsets, skip and incomplete-build recovery", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", action="store_true")
    args = parser.parse_args()
    try:
        if args.unit:
            unit_checks()
            return 0
        for chunker in CHUNKERS:
            check_stores(chunker)
        quality_checks()
        print("PASS index acceptance checks", flush=True)
        return 0
    except Exception as error:
        print(f"FAIL {type(error).__name__}: {error}", flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
