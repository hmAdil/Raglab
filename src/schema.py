from dataclasses import dataclass


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    score: float
    rank: int


class Retriever:
    config_id: str

    def retrieve(self, query: str, k: int = 5) -> list[Hit]:
        raise NotImplementedError


class Scorer:
    scorer_id: str

    def score(self, query: str, chunk_ids: list[str]) -> list[float]:
        raise NotImplementedError
