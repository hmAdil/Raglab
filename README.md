# RAGLab

Benchmarking RAG configurations for technical documentation question answering.
Course: UE24AM343AB5, Prompt Engineering and Optimization for Generative AI.

## Setup

python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

## Decisions

- Vector store: FAISS
- Generator and judge: served locally via Ollama or a free-tier API

## Data

data/raw is gitignored. Download the corpus sources listed below into data/raw.
Cleaned text goes in data/processed.

## Test set

data/testset.jsonl, one JSON object per line. Frozen at the start of week 2.
Validate with: python src/validate_testset.py

## Run instructions

To be added.

Python version: 3.10.11

Python version: 3.10.11
