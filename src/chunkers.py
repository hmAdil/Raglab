
import argparse
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np

from config import CHUNKERS, CHUNKS_DIR, FIXED_OVERLAP_TOKENS, MAX_CHUNK_TOKENS, RECURSIVE_OVERLAP_TOKENS, SEMANTIC_PERCENTILE, SENTENCES_PER_CHUNK, SOURCES, TOKENIZER_NAME
from data_access import load_passages


SENT_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`\"'(\[])")


@lru_cache(maxsize=1)
def get_tokenizer() -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    tokenizer.model_max_length = 10**12
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for exact character offsets")
    return tokenizer


def count_tokens(text: str) -> int:
    return len(get_tokenizer()(text, add_special_tokens=False)["input_ids"])


def window_chunks(token_offsets: list[tuple[int, int]], size: int, overlap: int) -> list[tuple[int, int, int]]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Window requires 0 <= overlap < size")
    n = len(token_offsets)
    step = size - overlap
    out = []
    i = 0
    while i < n:
        j = min(i + size, n)
        out.append((token_offsets[i][0], token_offsets[j - 1][1], j - i))
        if j == n:
            break
        i += step
    return out


def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    for line in re.finditer(r"[^\n]+", text):
        line_start = line.start()
        line_text = line.group(0)
        pos = 0
        for match in SENT_BREAK.finditer(line_text):
            spans.append((line_start + pos, line_start + match.start()))
            pos = match.end()
        spans.append((line_start + pos, line_start + len(line_text)))
    cleaned = []
    for a, b in spans:
        segment = text[a:b]
        a2 = a + len(segment) - len(segment.lstrip())
        b2 = b - len(segment) + len(segment.rstrip())
        if b2 > a2:
            cleaned.append((a2, b2))
    return cleaned


def group_sentences(text: str, spans: list[tuple[int, int]], n_per_chunk: int, max_tokens: int, count_tokens: Callable[[str], int], hard_split: Callable[[int, int], list[tuple[int, int]]]) -> list[tuple[int, int]]:
    chunks = []
    cur = []
    for sp in spans:
        n_tok = count_tokens(text[sp[0]:sp[1]])
        if n_tok > max_tokens:
            if cur:
                chunks.append((cur[0][0], cur[-1][1]))
                cur = []
            chunks.extend(hard_split(sp[0], sp[1]))
            continue
        trial = cur + [sp]
        too_long = count_tokens(text[trial[0][0]:trial[-1][1]]) > max_tokens
        if cur and (len(trial) > n_per_chunk or too_long):
            chunks.append((cur[0][0], cur[-1][1]))
            cur = [sp]
        else:
            cur = trial
    if cur:
        chunks.append((cur[0][0], cur[-1][1]))
    return chunks


def chunk_fixed_spans(text: str, tokenizer: Any, size: int, overlap: int) -> list[tuple[int, int]]:
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    if not offsets:
        return []
    return [(a, b) for a, b, _ in window_chunks(offsets, size, overlap)]


def bounded_fixed_spans(text: str, overlap: int) -> list[tuple[int, int]]:
    tokenizer = get_tokenizer()
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    output = []
    i = 0
    while i < len(offsets):
        j = min(i + MAX_CHUNK_TOKENS, len(offsets))
        while j > i and count_tokens(text[offsets[i][0]:offsets[j - 1][1]]) > MAX_CHUNK_TOKENS:
            j -= 1
        if j == i:
            raise ValueError("A token cannot be represented within the chunk cap")
        output.append((offsets[i][0], offsets[j - 1][1]))
        if j == len(offsets):
            break
        i = max(i + 1, j - overlap)
    return output


def hard_split(text: str, start: int, end: int) -> list[tuple[int, int]]:
    return [(start + a, start + b) for a, b in bounded_fixed_spans(text[start:end], 0)]


def semantic_spans(text: str, spans: list[tuple[int, int]], similarities: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    out = []
    current: list[tuple[int, int]] = []
    for i, span in enumerate(spans):
        if count_tokens(text[span[0]:span[1]]) > MAX_CHUNK_TOKENS:
            if current:
                out.append((current[0][0], current[-1][1]))
                current = []
            out.extend(hard_split(text, *span))
            continue
        boundary = i > 0 and similarities[i - 1] < threshold
        too_long = bool(current) and count_tokens(text[current[0][0]:span[1]]) > MAX_CHUNK_TOKENS
        if current and (boundary or too_long):
            out.append((current[0][0], current[-1][1]))
            current = []
        current.append(span)
    if current:
        out.append((current[0][0], current[-1][1]))
    return out


def locate_recursive_piece(text: str, piece: str, suggested: int, previous_start: int, previous_end: int) -> int:
    def valid(start: int) -> bool:
        return start >= previous_start and text[start:start + len(piece)] == piece and start + len(piece) > previous_end and (start <= previous_end or not text[previous_end:start].strip())

    if suggested >= 0 and valid(suggested):
        return suggested
    start = text.find(piece, previous_start)
    while start >= 0:
        if valid(start):
            return start
        start = text.find(piece, start + 1)
    raise ValueError("Recursive chunk cannot be located without losing source coverage")


@lru_cache(maxsize=1)
def semantic_source_data(source: str) -> tuple[list[list[tuple[int, int]]], list[np.ndarray], float]:
    from embedders import SplitEmbedder

    passages = load_passages(source)
    spans_by_passage = [sentence_spans(passage["text"]) for passage in passages]
    model = SplitEmbedder()
    sentences = [passage["text"][a:b] for passage, spans in zip(passages, spans_by_passage) for a, b in spans]
    print(f"Embedding semantic sentence units: {source}, {len(sentences)} units", flush=True)
    vectors = model.encode(sentences)
    print(f"Encoded semantic sentence units: {source}", flush=True)
    similarities = []
    offset = 0
    for spans in spans_by_passage:
        passage_vectors = vectors[offset:offset + len(spans)]
        similarities.append(np.sum(passage_vectors[:-1] * passage_vectors[1:], axis=1))
        offset += len(spans)
    joined = [values for values in similarities if len(values)]
    threshold = float(np.percentile(np.concatenate(joined), SEMANTIC_PERCENTILE)) if joined else float("-inf")
    print(f"Semantic threshold {source}: {threshold:.6f}", flush=True)
    return spans_by_passage, similarities, threshold


def chunk_passage(chunker: str, text: str) -> list[tuple[int, int]]:
    if chunker == "fixed":
        return bounded_fixed_spans(text, FIXED_OVERLAP_TOKENS)
    if chunker == "sentence":
        return group_sentences(text, sentence_spans(text), SENTENCES_PER_CHUNK, MAX_CHUNK_TOKENS, count_tokens, lambda a, b: hard_split(text, a, b))
    if chunker == "recursive":
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_CHUNK_TOKENS, chunk_overlap=RECURSIVE_OVERLAP_TOKENS, length_function=count_tokens, separators=["\n\n", "\n", ". ", " ", ""], add_start_index=True, strip_whitespace=True)
        out = []
        previous_start = 0
        previous_end = 0
        for document in splitter.create_documents([text]):
            piece = document.page_content
            start = locate_recursive_piece(text, piece, document.metadata["start_index"], previous_start, previous_end)
            end = start + len(piece)
            assert text[start:end] == piece
            out.append((start, end))
            previous_start = start
            previous_end = end
        return out
    if chunker == "semantic":
        for source in SOURCES:
            passages = load_passages(source)
            for i, passage in enumerate(passages):
                if passage["text"] == text:
                    spans, similarities, threshold = semantic_source_data(source)
                    return semantic_spans(text, spans[i], similarities[i], threshold)
        raise ValueError("Semantic chunking requires a corpus passage to select its source-wide threshold")
    raise ValueError(f"Unknown chunker: {chunker}")


def build_chunks(chunker: str) -> Path:
    if chunker not in CHUNKERS:
        raise ValueError(f"Unknown chunker: {chunker}")
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    path = CHUNKS_DIR / f"{chunker}.jsonl"
    temporary = path.with_suffix(".jsonl.tmp")
    counts = []
    with temporary.open("w", encoding="utf-8") as handle:
        for source in SOURCES:
            passages = load_passages(source)
            semantic = semantic_source_data(source) if chunker == "semantic" else None
            for passage_idx, passage in enumerate(passages):
                text = passage["text"]
                if semantic is None:
                    spans = chunk_passage(chunker, text)
                else:
                    groups, similarities, threshold = semantic
                    spans = semantic_spans(text, groups[passage_idx], similarities[passage_idx], threshold)
                for start, end in spans:
                    piece = text[start:end]
                    n_tokens = count_tokens(piece)
                    if not 1 <= n_tokens <= MAX_CHUNK_TOKENS:
                        raise ValueError(f"Invalid chunk size {n_tokens}: {source}/{passage_idx}/{start}:{end}")
                    record = {"chunk_id": f"{chunker}-{len(counts):06d}", "chunker": chunker, "source": source, "passage_idx": passage_idx, "section": passage["section"], "url": passage["url"], "char_start": start, "char_end": end, "n_bert_tokens": n_tokens, "text": piece}
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    counts.append(n_tokens)
    temporary.replace(path)
    stats_path = CHUNKS_DIR / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    stats[chunker] = {"n_chunks": len(counts), "mean": float(np.mean(counts)), "median": float(np.median(counts)), "max": max(counts)}
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"Built {chunker}: {len(counts)} chunks, max {max(counts)} BERT tokens", flush=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunker", choices=["all", *CHUNKERS], default="all")
    args = parser.parse_args()
    selected = ("fixed", "sentence", "recursive", "semantic") if args.chunker == "all" else (args.chunker,)
    for chunker in selected:
        build_chunks(chunker)
    print(f"Built {len(selected)} chunk files", flush=True)


if __name__ == "__main__":
    main()
