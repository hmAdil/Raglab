import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
from typing import Any

try:
    from config import SEED
except ImportError:
    SEED = 42

try:
    from schema import Hit, Retriever
except ImportError:
    @dataclass(frozen=True)
    class Hit:
        chunk_id: str
        score: float
        rank: int

    class Retriever:
        config_id: str

        def retrieve(self, query: str, k: int = 5) -> list[Hit]:
            raise NotImplementedError

try:
    from data_access import load_passages, load_testset
except ImportError:
    ROOT = Path(__file__).resolve().parent.parent

    def load_passages(source: str) -> list[dict[str, Any]]:
        path = ROOT / "data" / "processed" / f"{source}.jsonl"
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def load_testset() -> list[dict[str, Any]]:
        path = ROOT / "data" / "testset.jsonl"
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

def _normalize_section(s: str | None) -> str:
    val = (s or "").replace("\ufffd", "").replace("\u00b6", "").replace("#", "")
    return re.sub(r"\s+", " ", val).strip().lower()

class OracleRetriever(Retriever):
    config_id: str = "oracle"

    def __init__(self, seed: int = SEED) -> None:
        self.seed = seed
        self.passages: dict[str, list[dict[str, Any]]] = {}
        for src in ("fastapi", "pytorch"):
            self.passages[src] = load_passages(src)
        self.testset = load_testset()
        self.query_to_question: dict[str, dict[str, Any]] = {
            q["question"].strip().lower(): q for q in self.testset
        }
        self.chunk_cache: dict[str, dict[str, Any]] = {}
        self._build_chunk_cache()

    def _build_chunk_cache(self) -> None:
        for src, plist in self.passages.items():
            for idx, p in enumerate(plist):
                cid = f"oracle-{src}-{idx}"
                self.chunk_cache[cid] = {
                    "chunk_id": cid,
                    "chunker": "oracle",
                    "source": src,
                    "passage_idx": idx,
                    "section": p.get("section", ""),
                    "url": p.get("url", ""),
                    "char_start": 0,
                    "char_end": len(p.get("text", "")),
                    "n_bert_tokens": p.get("token_count", len(p.get("text", "").split())),
                    "text": p.get("text", ""),
                }

    def get_chunk_record(self, chunk_id: str) -> dict[str, Any]:
        return self.chunk_cache[chunk_id]

    def _resolve_gold_passage(self, gold: dict[str, Any]) -> tuple[str, int]:
        source = gold["source"]
        span = gold["span"]
        sec = _normalize_section(gold.get("section"))
        plist = self.passages.get(source, [])
        for idx, p in enumerate(plist):
            if _normalize_section(p.get("section")) == sec and span in p.get("text", ""):
                return source, idx
        for idx, p in enumerate(plist):
            if span in p.get("text", ""):
                return source, idx
        return source, 0

    def retrieve_for_question(self, question_dict: dict[str, Any], k: int = 5) -> list[Hit]:
        rng = random.Random(self.seed)
        is_answerable = bool(question_dict.get("answerable", False))
        if is_answerable and "gold_evidence" in question_dict:
            src, gold_idx = self._resolve_gold_passage(question_dict["gold_evidence"])
            plist = self.passages[src]
            other_indices = [i for i in range(len(plist)) if i != gold_idx]
            rng.shuffle(other_indices)
            chosen_indices = [gold_idx] + other_indices[: k - 1]
            hits: list[Hit] = []
            score_step = 1.0 / float(k)
            for rank_idx, pidx in enumerate(chosen_indices, start=1):
                cid = f"oracle-{src}-{pidx}"
                score = 1.0 - (rank_idx - 1) * score_step
                hits.append(Hit(chunk_id=cid, score=float(score), rank=rank_idx))
            return hits
        else:
            default_src = "fastapi"
            plist = self.passages[default_src]
            all_indices = list(range(len(plist)))
            rng.shuffle(all_indices)
            chosen_indices = all_indices[:k]
            hits: list[Hit] = []
            score_step = 1.0 / float(k)
            for rank_idx, pidx in enumerate(chosen_indices, start=1):
                cid = f"oracle-{default_src}-{pidx}"
                score = 1.0 - (rank_idx - 1) * score_step
                hits.append(Hit(chunk_id=cid, score=float(score), rank=rank_idx))
            return hits

    def retrieve(self, query: str, k: int = 5) -> list[Hit]:
        q_norm = query.strip().lower()
        if q_norm in self.query_to_question:
            return self.retrieve_for_question(self.query_to_question[q_norm], k=k)
        default_src = "fastapi"
        plist = self.passages[default_src]
        rng = random.Random(self.seed)
        all_indices = list(range(len(plist)))
        rng.shuffle(all_indices)
        chosen_indices = all_indices[:k]
        hits: list[Hit] = []
        score_step = 1.0 / float(k)
        for rank_idx, pidx in enumerate(chosen_indices, start=1):
            cid = f"oracle-{default_src}-{pidx}"
            score = 1.0 - (rank_idx - 1) * score_step
            hits.append(Hit(chunk_id=cid, score=float(score), rank=rank_idx))
        return hits

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    retriever = OracleRetriever()
    tset = retriever.testset
    if tset:
        sample_q = tset[0]
        hits = retriever.retrieve_for_question(sample_q, k=args.k)
        print(f"Retrieved {len(hits)} hits for question {sample_q.get('id')}:")
        for h in hits:
            print(f"  rank={h.rank} chunk_id={h.chunk_id} score={h.score:.3f}")

if __name__ == "__main__":
    main()
