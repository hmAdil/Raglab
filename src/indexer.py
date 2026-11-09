
import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from config import CHUNKERS, EMBEDDERS, INDEX_DIR, RESCORING_DIR, SINGLE_VECTOR_MODELS
from data_access import load_chunks
from embedders import ColBertEmbedder, SentenceEmbedder


def artifact_path(chunker: str, model: str) -> Path:
    return (INDEX_DIR if model in EMBEDDERS else RESCORING_DIR) / f"{chunker}__{model}"


def complete_artifact(path: Path, model: str, ids: list[str]) -> bool:
    required = ["index.faiss"] if model in EMBEDDERS else ["vectors.npy"] if model == "distilbert" else ["tokens.npy", "offsets.npy"]
    if not all((path / name).is_file() for name in ["chunk_ids.json", *required]):
        return False
    try:
        with (path / "chunk_ids.json").open(encoding="utf-8") as handle:
            if json.load(handle) != ids:
                return False
        if model in EMBEDDERS:
            import faiss

            index = faiss.read_index(str(path / "index.faiss"))
            return index.ntotal == len(ids) and index.d == SINGLE_VECTOR_MODELS[model]["dim"]
        if model == "distilbert":
            vectors = np.load(path / "vectors.npy", mmap_mode="r")
            return vectors.shape == (len(ids), SINGLE_VECTOR_MODELS[model]["dim"]) and vectors.dtype == np.float32
        tokens = np.load(path / "tokens.npy", mmap_mode="r")
        offsets = np.load(path / "offsets.npy")
        return tokens.ndim == 2 and tokens.shape[1] == 128 and tokens.dtype == np.float16 and offsets.shape == (len(ids) + 1,) and offsets.dtype == np.int64 and offsets[0] == 0 and offsets[-1] == len(tokens) and bool(np.all(np.diff(offsets) > 0))
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
        return False


def build_index(chunker: str, model: str, force: bool = False) -> Path:
    if chunker not in CHUNKERS or model not in (*EMBEDDERS, "distilbert", "colbert"):
        raise ValueError(f"Unknown artifact: {chunker}/{model}")
    started = time.perf_counter()
    chunks = load_chunks(chunker)
    if not chunks:
        raise ValueError(f"No chunks for {chunker}")
    ids = [chunk["chunk_id"] for chunk in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate chunk IDs")
    target = artifact_path(chunker, model)
    if not force and complete_artifact(target, model, ids):
        print(f"Skipped {chunker}__{model}: complete artifact for {len(ids)} chunks", flush=True)
        return target
    target.mkdir(parents=True, exist_ok=True)
    order = sorted(range(len(chunks)), key=lambda i: (chunks[i]["n_bert_tokens"], i))
    with tempfile.TemporaryDirectory(prefix=".build-", dir=target) as temporary:
        stage = Path(temporary)
        if model == "colbert":
            encoder = ColBertEmbedder()
            lengths = np.zeros(len(chunks), dtype=np.int64)
            for start in range(0, len(order), 16):
                rows = order[start:start + 16]
                values = encoder.encode_documents([chunks[i]["text"] for i in rows])
                for row, value in zip(rows, values):
                    lengths[row] = len(value)
                    np.save(stage / f"row-{row}.npy", value.astype(np.float16))
            offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(lengths, dtype=np.int64)])
            tokens = np.lib.format.open_memmap(stage / "tokens.npy", mode="w+", dtype=np.float16, shape=(int(offsets[-1]), 128))
            for row in range(len(chunks)):
                row_path = stage / f"row-{row}.npy"
                tokens[offsets[row]:offsets[row + 1]] = np.load(row_path)
                row_path.unlink()
            tokens.flush()
            del tokens
            np.save(stage / "offsets.npy", offsets)
            n_vectors = int(offsets[-1])
            names = ("tokens.npy", "offsets.npy")
        else:
            encoder = SentenceEmbedder(model)
            vectors = np.empty((len(chunks), SINGLE_VECTOR_MODELS[model]["dim"]), dtype=np.float32)
            batch = SINGLE_VECTOR_MODELS[model]["batch_size"]
            for start in range(0, len(order), batch):
                rows = order[start:start + batch]
                vectors[rows] = encoder.encode_documents([chunks[i]["text"] for i in rows])
            if model in EMBEDDERS:
                import faiss

                index = faiss.IndexFlatIP(SINGLE_VECTOR_MODELS[model]["dim"])
                index.add(np.ascontiguousarray(vectors))
                faiss.write_index(index, str(stage / "index.faiss"))
                names = ("index.faiss",)
            else:
                np.save(stage / "vectors.npy", vectors)
                names = ("vectors.npy",)
            n_vectors = len(vectors)
        (stage / "chunk_ids.json").write_text(json.dumps(ids) + "\n", encoding="utf-8")
        for name in (*names, "chunk_ids.json"):
            (stage / name).replace(target / name)
    print(f"Built {chunker}__{model}: {len(ids)} chunks, {n_vectors} vectors, {time.perf_counter() - started:.2f}s", flush=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunker", choices=["all", *CHUNKERS], default="all")
    parser.add_argument("--model", choices=["all", *EMBEDDERS, "distilbert", "colbert"], default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    chunkers = CHUNKERS if args.chunker == "all" else (args.chunker,)
    models = (*EMBEDDERS, "distilbert", "colbert") if args.model == "all" else (args.model,)
    for chunker in chunkers:
        for model in models:
            build_index(chunker, model, args.force)
    print(f"Finished {len(chunkers) * len(models)} artifact builds", flush=True)


if __name__ == "__main__":
    main()
