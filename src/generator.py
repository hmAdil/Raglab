import argparse
from typing import Any
from prompts import format_context, build_prompt
from llm_client import ollama_chat

def generate(style: str, question: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    context = format_context(chunks)
    prompt = build_prompt(style, question, context)
    chat_res = ollama_chat(prompt)
    return {
        "answer": chat_res["answer"],
        "prompt_tokens": chat_res["prompt_tokens"],
        "completion_tokens": chat_res["completion_tokens"],
        "generation_ms": chat_res["latency_ms"],
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", choices=["plain", "grounded"], default="plain")
    parser.add_argument("--question", default="How do I declare a request body?")
    args = parser.parse_args()
    sample_chunks = [
        {"chunk_id": "fastapi-000001", "text": "Use Pydantic models to declare request bodies in FastAPI."}
    ]
    res = generate(args.style, args.question, sample_chunks)
    print(res["answer"])

if __name__ == "__main__":
    main()
