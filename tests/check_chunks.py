import argparse
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chunkers import chunk_fixed_spans, count_tokens, group_sentences, sentence_spans, window_chunks
from config import CHUNKERS, COLBERT_MODEL, MAX_CHUNK_TOKENS, SEED, SINGLE_VECTOR_MODELS, SOURCES, TOKENIZER_NAME
from data_access import load_chunks, load_passages


def validate_chunks(chunker: str, records: list[dict], passages: dict[str, list[dict]], counter: Any) -> dict:
    coverage = {(source, i): np.zeros(len(passage["text"]), dtype=bool) for source, rows in passages.items() for i, passage in enumerate(rows)}
    counts = []
    previous = (-1, -1, -1)
    for i, chunk in enumerate(records):
        assert chunk["chunk_id"] == f"{chunker}-{i:06d}", chunk["chunk_id"]
        assert chunk["chunker"] == chunker
        source, pidx = chunk["source"], chunk["passage_idx"]
        assert source in passages and 0 <= pidx < len(passages[source])
        passage = passages[source][pidx]
        start, end = chunk["char_start"], chunk["char_end"]
        assert 0 <= start < end <= len(passage["text"])
        assert chunk["text"] == passage["text"][start:end], chunk["chunk_id"]
        assert chunk["section"] == passage["section"] and chunk["url"] == passage["url"]
        key = (SOURCES.index(source), pidx, start)
        assert key >= previous, f"Out-of-order chunk: {chunk['chunk_id']}"
        previous = key
        n = counter(chunk["text"])
        assert n == chunk["n_bert_tokens"] and 1 <= n <= MAX_CHUNK_TOKENS, chunk["chunk_id"]
        coverage[source, pidx][start:end] = True
        counts.append(n)
    assert len(records) >= sum(len(rows) for rows in passages.values())
    for (source, pidx), covered in coverage.items():
        missing = [i for i, char in enumerate(passages[source][pidx]["text"]) if not char.isspace() and not covered[i]]
        assert not missing, f"Uncovered characters: {source}/{pidx}: {missing[:10]}"
    return {"n_chunks": len(records), "mean": float(np.mean(counts)), "median": float(np.median(counts)), "max": max(counts)}


def check_tokenizers(passages: dict[str, list[dict]]) -> None:
    from transformers import AutoTokenizer

    samples = random.Random(SEED).sample([p["text"] for rows in passages.values() for p in rows], 50)
    expected = [count_tokens(text) for text in samples]
    for name in [settings["model"] for settings in SINGLE_VECTOR_MODELS.values()] + [COLBERT_MODEL]:
        tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
        tokenizer.model_max_length = 10**12
        actual = [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in samples]
        assert actual == expected, f"Tokenizer mismatch: {name} versus {TOKENIZER_NAME}; tell the team"
        print(f"PASS tokenizer counts: {name} (50 passages)", flush=True)


def unit_checks() -> None:
    class WordTokenizer:
        def __call__(self, text: str, **kwargs: Any) -> dict:
            return {"offset_mapping": [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]}

    assert window_chunks([(i, i + 1) for i in range(9)], 4, 1) == [(0, 4, 4), (3, 7, 4), (6, 9, 3)]
    for text in ["", " \n\t", " One sentence. Next one!\nCode(x)\n", " ".join(["token"] * 1001)]:
        spans = chunk_fixed_spans(text, WordTokenizer(), 400, 50)
        covered = set()
        for a, b in spans:
            assert len(text[a:b].split()) <= 400
            covered.update(range(a, b))
        assert all(char.isspace() or i in covered for i, char in enumerate(text))
    text = "One. Two! Three?\nFour. Five. Six."
    spans = sentence_spans(text)
    assert [text[a:b] for a, b in spans] == ["One.", "Two!", "Three?", "Four.", "Five.", "Six."]
    groups = group_sentences(text, spans, 5, 400, lambda value: len(value.split()), lambda a, b: [(a, b)])
    assert groups == [(spans[0][0], spans[4][1]), spans[5]]
    print("PASS offline window, coverage and sentence-group checks", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunker", choices=["all", *CHUNKERS], default="all")
    parser.add_argument("--unit", action="store_true")
    args = parser.parse_args()
    try:
        if args.unit:
            unit_checks()
            return 0
        passages = {source: load_passages(source) for source in SOURCES}
        selected = CHUNKERS if args.chunker == "all" else (args.chunker,)
        for chunker in selected:
            stats = validate_chunks(chunker, load_chunks(chunker), passages, count_tokens)
            print(f"{chunker:10} {stats['n_chunks']:6d} mean={stats['mean']:.2f} median={stats['median']:.2f} max={stats['max']}", flush=True)
        check_tokenizers(passages)
        print("PASS chunk acceptance checks", flush=True)
        return 0
    except Exception as error:
        print(f"FAIL {type(error).__name__}: {error}", flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
