import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
TESTSET_PATH = DATA_DIR / "testset.jsonl"
ARTIFACTS_DIR = ROOT / "artifacts"
CHUNKS_DIR = ARTIFACTS_DIR / "chunks"
INDEX_DIR = ARTIFACTS_DIR / "indexes"
RESCORING_DIR = ARTIFACTS_DIR / "rescoring"
BM25_DIR = ARTIFACTS_DIR / "bm25"
RESULTS_DIR = ROOT / "results"
CHARTS_DIR = RESULTS_DIR / "charts"
CACHE_DIR = ROOT / "cache"

SOURCES = ("fastapi", "pytorch")
CHUNKERS = ("fixed", "sentence", "semantic", "recursive")
EMBEDDERS = ("bge", "minilm")
RESCORERS = ("colbert", "distilbert")
RETRIEVALS = ("colbert", "distilbert", "colbert_bm25", "distilbert_bm25")
PROMPT_STYLES = ("plain", "grounded")

SEED = 42
TOP_K = 5
CANDIDATE_DEPTH = 20
RRF_K = 60
TOP_CONFIGS_FOR_STAGE2 = 3

TOKENIZER_NAME = "bert-base-uncased"
MAX_CHUNK_TOKENS = 400
FIXED_OVERLAP_TOKENS = 50
RECURSIVE_OVERLAP_TOKENS = 50
SENTENCES_PER_CHUNK = 5
SEMANTIC_PERCENTILE = 20

SEMANTIC_SPLIT_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDER_MAX_SEQ = 512
SINGLE_VECTOR_MODELS = {
    "bge": {
        "model": "BAAI/bge-small-en-v1.5",
        "dim": 384,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "batch_size": 64,
    },
    "minilm": {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
        "query_prefix": "",
        "batch_size": 64,
    },
    "distilbert": {
        "model": "sentence-transformers/multi-qa-distilbert-cos-v1",
        "dim": 768,
        "query_prefix": "",
        "batch_size": 32,
    },
}
COLBERT_MODEL = "colbert-ir/colbertv2.0"
COLBERT_LOADER = "sentence_transformers"
COLBERT_DOC_TOKENS = 512
COLBERT_QUERY_TOKENS = 32

GEN_BASE_URL = os.environ.get("GEN_BASE_URL", "http://localhost:11434")
GEN_MODEL = os.environ.get("GEN_MODEL", "llama3.2:3b")
GEN_NUM_CTX = 8192
GEN_MAX_TOKENS = 400
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "")
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", "")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "")
JUDGE_MAX_TOKENS = 300


def config_id(chunker: str, embedder: str, retrieval: str) -> str:
    return f"{chunker}__{embedder}__{retrieval}"


def parse_retrieval(retrieval: str) -> tuple[str, bool]:
    return retrieval.split("_")[0], retrieval.endswith("_bm25")

