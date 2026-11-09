import json
from pathlib import Path

from config import CHUNKS_DIR, PROCESSED_DIR, TESTSET_PATH


def read_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_passages(source: str) -> list[dict]:
    return read_jsonl(PROCESSED_DIR / f"{source}.jsonl")


def load_chunks(chunker: str) -> list[dict]:
    return read_jsonl(CHUNKS_DIR / f"{chunker}.jsonl")


def chunk_index(chunker: str) -> dict[str, dict]:
    return {c["chunk_id"]: c for c in load_chunks(chunker)}


def load_testset() -> list[dict]:
    return read_jsonl(TESTSET_PATH)
