import json
from pathlib import Path

from config import CHUNKERS, CHUNKS_DIR, PROCESSED_DIR, SOURCES, TESTSET_PATH


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_passages(source: str) -> list[dict]:
    if source not in SOURCES:
        raise ValueError(f"Unknown source: {source}")
    return read_jsonl(PROCESSED_DIR / f"{source}.jsonl")


def load_chunks(chunker: str) -> list[dict]:
    if chunker not in CHUNKERS:
        raise ValueError(f"Unknown chunker: {chunker}")
    return read_jsonl(CHUNKS_DIR / f"{chunker}.jsonl")


def chunk_index(chunker: str) -> dict[str, dict]:
    records = load_chunks(chunker)
    lookup = {record["chunk_id"]: record for record in records}
    if len(lookup) != len(records):
        raise ValueError(f"Duplicate chunk IDs in {chunker}")
    return lookup


def load_testset() -> list[dict]:
    return read_jsonl(TESTSET_PATH)
