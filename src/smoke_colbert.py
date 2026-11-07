import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from config import COLBERT_DOC_TOKENS, COLBERT_QUERY_TOKENS


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def set_lengths(model: Any, loader: str) -> None:
    if loader == "pylate":
        model.document_length = COLBERT_DOC_TOKENS
        model.query_length = COLBERT_QUERY_TOKENS
    for module in model.modules():
        if hasattr(module, "document_length"):
            module.document_length = COLBERT_DOC_TOKENS
        if hasattr(module, "query_length"):
            module.query_length = COLBERT_QUERY_TOKENS


def encode(model: Any, loader: str, texts: list[str], query: bool) -> list[np.ndarray]:
    import torch

    with torch.inference_mode():
        if loader == "pylate":
            values = model.encode(texts, is_query=query, batch_size=16, show_progress_bar=False)
        else:
            method = model.encode_query if query else model.encode_document
            values = method(texts, batch_size=16, show_progress_bar=False)
    return [as_numpy(value) for value in values]


def maxsim_scores(q: np.ndarray, tokens: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    sim = q @ tokens.T
    per_chunk_max = np.maximum.reduceat(sim, offsets[:-1], axis=1)
    return per_chunk_max.sum(axis=0)


def verify(model: Any, loader: str) -> None:
    import torch
    from transformers import AutoTokenizer

    set_lengths(model, loader)
    q = encode(model, loader, ["how do I declare a request body?"], True)
    d = encode(model, loader, ["Use Pydantic models to declare a request body.", "A tensor is a multidimensional array."], False)
    assert all(v.ndim == 2 and v.shape[1] == 128 and len(v) for v in q + d)
    print("PASS token embedding dimensions", flush=True)
    qn = [v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12) for v in q]
    dn = [v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12) for v in d]
    scores = maxsim_scores(qn[0], np.concatenate(dn), np.asarray([0, len(dn[0]), sum(map(len, dn))]))
    assert scores[0] > scores[1]
    print("PASS request-body ranking", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    long_text = " ".join(["request"] * 450)
    assert len(tokenizer(long_text, add_special_tokens=False)["input_ids"]) == 450
    long_vectors = encode(model, loader, [long_text], False)[0]
    assert 400 <= len(long_vectors) <= COLBERT_DOC_TOKENS, long_vectors.shape
    print(f"PASS document length: {len(long_vectors)} vectors for 450 BERT tokens", flush=True)
    norms = np.concatenate([np.linalg.norm(v, axis=1) for v in q + d])
    print("PASS vector norms" if np.allclose(norms, 1, atol=0.01) else "PASS vector norms: wrapper normalisation required", flush=True)
    if loader == "pylate":
        from pylate import scores as pylate_scores

        native = as_numpy(pylate_scores.colbert_scores(q, d)).reshape(-1)
    else:
        with torch.inference_mode():
            native = as_numpy(model.similarity([torch.from_numpy(v) for v in q], [torch.from_numpy(v) for v in d])).reshape(-1)
    assert np.allclose(scores, native, atol=0.05), (scores, native)
    print("PASS MaxSim agreement", flush=True)


def load_candidate(name: str, loader: str) -> Any:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if loader == "sentence_transformers":
        from sentence_transformers import MultiVectorEncoder

        return MultiVectorEncoder(name, device=device)
    from pylate import models

    return models.ColBERT(model_name_or_path=name, device=device)


def prepare_pylate() -> bool:
    report = Path(__file__).resolve().parent.parent / "cache" / "pylate_install_plan.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([sys.executable, "-m", "pip", "install", "pylate", "--dry-run", "--report", str(report)], check=False)
    if result.returncode:
        return False
    with report.open(encoding="utf-8") as handle:
        plan = json.load(handle)
    protected = {"torch", "transformers", "sentence-transformers"}
    changed = {item["metadata"]["name"].lower().replace("_", "-") for item in plan.get("install", [])}
    if protected & changed:
        print(f"FAIL PyLate would change protected packages: {sorted(protected & changed)}", flush=True)
        return False
    subprocess.run([sys.executable, "-m", "pip", "install", "pylate"], check=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    missing = [name for name in ("torch", "transformers", "sentence_transformers") if importlib.util.find_spec(name) is None]
    if missing:
        print(f"FAIL missing dependencies: {', '.join(missing)}", flush=True)
        return 1
    attempts = [("colbert-ir/colbertv2.0", "sentence_transformers"), ("lightonai/colbertv2.0", "sentence_transformers")]
    for name, loader in attempts:
        try:
            model = load_candidate(name, loader)
            verify(model, loader)
        except Exception as error:
            print(f"FAIL {name} ({loader}): {type(error).__name__}: {error}", flush=True)
            continue
        record_choice(name, loader)
        return 0
    if prepare_pylate():
        try:
            model = load_candidate("lightonai/colbertv2.0", "pylate")
            verify(model, "pylate")
        except Exception as error:
            print(f"FAIL PyLate: {type(error).__name__}: {error}", flush=True)
        else:
            record_choice("lightonai/colbertv2.0", "pylate")
            return 0
    print("FAIL no verified loader; tell the team before using answerdotai/answerai-colbert-small-v1", flush=True)
    return 1


def record_choice(name: str, loader: str) -> None:
    import re

    path = Path(__file__).with_name("config.py")
    text = path.read_text(encoding="utf-8")
    for key, value in (("COLBERT_MODEL", name), ("COLBERT_LOADER", loader)):
        text = re.sub(rf"^{key} = .*?$", f"{key} = {json.dumps(value)}", text, flags=re.MULTILINE)
    path.write_text(text, encoding="utf-8")
    print(f"PASS ColBERT: {name} ({loader})", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
