
from typing import Any

import numpy as np

from config import COLBERT_LOADER, COLBERT_MODEL, EMBEDDER_MAX_SEQ, SEED, SEMANTIC_SPLIT_MODEL, SINGLE_VECTOR_MODELS
from smoke_colbert import as_numpy, encode, load_candidate, set_lengths


def seed_all() -> None:
    import random
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected a finite two-dimensional embedding array")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Zero embedding vector")
    return np.asarray(values / norms, dtype=np.float32)


class SentenceEmbedder:
    def __init__(self, model_id: str) -> None:
        if model_id not in SINGLE_VECTOR_MODELS:
            raise ValueError(f"Unknown single-vector model: {model_id}")
        self.model_id = model_id
        self.settings = SINGLE_VECTOR_MODELS[model_id]
        self._model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            seed_all()
            self._model = SentenceTransformer(self.settings["model"], device="cuda" if torch.cuda.is_available() else "cpu")
            self._model.max_seq_length = EMBEDDER_MAX_SEQ
            self._model.eval()
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        import torch

        if not texts:
            return np.empty((0, self.settings["dim"]), dtype=np.float32)
        with torch.inference_mode():
            values = self.model.encode(texts, batch_size=self.settings["batch_size"], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        result = normalize_rows(values)
        if result.shape != (len(texts), self.settings["dim"]):
            raise ValueError(f"Unexpected {self.model_id} shape: {result.shape}")
        return result

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        prefix = self.settings["query_prefix"]
        return self._encode([prefix + text for text in texts])


class ColBertEmbedder:
    def __init__(self) -> None:
        self._model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            seed_all()
            self._model = load_candidate(COLBERT_MODEL, COLBERT_LOADER)
            set_lengths(self._model, COLBERT_LOADER)
            self._model.eval()
        return self._model

    def encode_documents(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        result = [normalize_rows(value) for value in encode(self.model, COLBERT_LOADER, texts, False)]
        if len(result) != len(texts) or any(value.shape[1] != 128 or not len(value) for value in result):
            raise ValueError("Invalid ColBERT document embeddings")
        return result

    def encode_query(self, text: str) -> np.ndarray:
        result = normalize_rows(encode(self.model, COLBERT_LOADER, [text], True)[0])
        if result.shape[1] != 128 or not len(result):
            raise ValueError("Invalid ColBERT query embeddings")
        return result


class SplitEmbedder:
    def __init__(self) -> None:
        self._model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            seed_all()
            self._model = SentenceTransformer(SEMANTIC_SPLIT_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
            self._model.eval()
        return self._model

    def encode(self, sentences: list[str]) -> np.ndarray:
        import torch

        if not sentences:
            return np.empty((0, 768), dtype=np.float32)
        with torch.inference_mode():
            values = self.model.encode(sentences, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return normalize_rows(as_numpy(values))
