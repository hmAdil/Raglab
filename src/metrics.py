import re

import numpy as np


def normalize_section(s: str | None) -> str:
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
    for idx, p in enumerate(passages):
        if normalize_section(p["section"]) == sec:
            for a, b in find_all(p["text"], span):
                locs.append((idx, a, b))
    if locs:
        return locs
    for idx, p in enumerate(passages):
        for a, b in find_all(p["text"], span):
            locs.append((idx, a, b))
    return locs


def score_retrieval(retrieved_chunks: list[dict], locs: list[tuple[int, int, int]]) -> dict:
    best_cov = 0.0
    hit = False
    for (pidx, ga, gb) in locs:
        covered = np.zeros(gb - ga, dtype=bool)
        for c in retrieved_chunks:
            if c["passage_idx"] != pidx:
                continue
            a = max(ga, c["char_start"])
            b = min(gb, c["char_end"])
            if b > a:
                covered[a - ga:b - ga] = True
        cov = covered.mean()
        best_cov = max(best_cov, cov)
        hit = hit or cov > 0
    n_rel = 0
    for c in retrieved_chunks:
        for (pidx, ga, gb) in locs:
            if c["passage_idx"] == pidx and min(gb, c["char_end"]) > max(ga, c["char_start"]):
                n_rel += 1
                break
    precision = n_rel / len(retrieved_chunks) if retrieved_chunks else 0.0
    return {"context_recall": float(best_cov), "hit": bool(hit), "context_precision": precision}
