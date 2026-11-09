import argparse
from typing import Any

def format_context(chunks: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for chunk in chunks:
        cid = chunk["chunk_id"]
        text = chunk["text"]
        blocks.append(f"[{cid}] {text}")
    return "\n\n".join(blocks)

def build_prompt(style: str, question: str, context: str) -> str:
    if style == "plain":
        return f"Answer the question using the context.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    elif style == "grounded":
        return f"Answer the question using only the context below. Cite the chunk IDs you used in square brackets, for example [fixed-000123]. If the context does not contain the answer, reply exactly: Not found in the provided documents.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    raise ValueError(f"Unsupported prompt style: {style}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", choices=["plain", "grounded"], default="plain")
    parser.add_argument("--question", default="What is FastAPI?")
    args = parser.parse_args()
    sample_chunks = [{"chunk_id": "test-000001", "text": "FastAPI is a modern web framework."}]
    ctx = format_context(sample_chunks)
    prompt = build_prompt(args.style, args.question, ctx)
    print(prompt)

if __name__ == "__main__":
    main()
