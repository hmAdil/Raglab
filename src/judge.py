import argparse
import json
import os
from typing import Any
from llm_client import openai_chat

try:
    from config import JUDGE_BASE_URL, JUDGE_MODEL
except ImportError:
    JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "")
    JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "")

SYSTEM_PROMPT = "You are a strict evaluator. Output only valid JSON."

ANSWERABLE_USER_TEMPLATE = """Question:
{question}

Reference answer:
{reference_answer}

Retrieved context:
{context}

System answer:
{answer}

Score the system answer on two criteria, each an integer from 1 to 5.

Groundedness: are all claims in the system answer supported by the retrieved context?
5 = every claim is supported. 4 = one minor unsupported detail. 3 = some claims are
unsupported. 2 = most claims are unsupported. 1 = the answer contradicts or ignores the
context. An answer that only says the information was not found is grounded (5).

Correctness: does the system answer agree with the reference answer?
5 = all key facts match. 4 = correct with a minor omission or addition. 3 = partly
correct. 2 = mostly incorrect. 1 = incorrect, contradictory, or says the information was
not found.

Return JSON with exactly these keys:
{{"groundedness": <int 1-5>, "correctness": <int 1-5>, "justification": "<one or two sentences>"}}"""

UNANSWERABLE_USER_TEMPLATE = """Question:
{question}

The documents do not contain the answer to this question.

System answer:
{answer}

Did the system answer clearly say that it cannot answer or that the information is not
available, without inventing an answer?

Return JSON with exactly these keys:
{{"refused": <true or false>, "justification": "<one sentence>"}}"""

def _clean_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
            cleaned = "\n".join(lines[1:-1]).strip()
    return cleaned

def _parse_judge_json(raw: str, expected_type: str) -> dict[str, Any]:
    cleaned = _clean_code_fences(raw)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in response")
    snippet = cleaned[start:end + 1]
    data = json.loads(snippet)
    if not isinstance(data, dict):
        raise ValueError("Parsed JSON is not a dictionary")

    if expected_type == "answerable":
        for k in ("groundedness", "correctness", "justification"):
            if k not in data:
                raise ValueError(f"Missing key in answerable judgment: {k}")
        g = data["groundedness"]
        c = data["correctness"]
        if isinstance(g, bool) or not isinstance(g, int) or g < 1 or g > 5:
            raise ValueError(f"Invalid groundedness value: {g}")
        if isinstance(c, bool) or not isinstance(c, int) or c < 1 or c > 5:
            raise ValueError(f"Invalid correctness value: {c}")
        return {
            "groundedness": int(g),
            "correctness": int(c),
            "justification": str(data["justification"]),
        }
    elif expected_type == "refusal":
        for k in ("refused", "justification"):
            if k not in data:
                raise ValueError(f"Missing key in refusal judgment: {k}")
        r = data["refused"]
        if not isinstance(r, bool):
            raise ValueError(f"Invalid refused value: {r}")
        return {
            "refused": bool(r),
            "justification": str(data["justification"]),
        }
    raise ValueError(f"Unknown expected type: {expected_type}")

def _run_judge_with_retry(messages: list[dict[str, Any]], expected_type: str) -> dict[str, Any]:
    if not JUDGE_BASE_URL or not JUDGE_MODEL:
        err = "Judge configuration missing (JUDGE_BASE_URL or JUDGE_MODEL not set)"
        if expected_type == "answerable":
            return {"groundedness": None, "correctness": None, "justification": None, "judge_error": err}
        return {"refused": None, "justification": None, "judge_error": err}

    try:
        res = openai_chat(messages)
        parsed = _parse_judge_json(res["content"], expected_type)
        parsed["judge_error"] = None
        return parsed
    except Exception as exc1:
        retry_messages = list(messages)
        retry_messages.append({"role": "user", "content": "Return only the JSON object."})
        try:
            res_retry = openai_chat(retry_messages)
            parsed_retry = _parse_judge_json(res_retry["content"], expected_type)
            parsed_retry["judge_error"] = None
            return parsed_retry
        except Exception as exc2:
            err_msg = f"{type(exc2).__name__}: {str(exc2)}"
            if expected_type == "answerable":
                return {"groundedness": None, "correctness": None, "justification": None, "judge_error": err_msg}
            return {"refused": None, "justification": None, "judge_error": err_msg}

def judge_answerable(question: str, reference_answer: str, context: str, answer: str) -> dict[str, Any]:
    user_prompt = ANSWERABLE_USER_TEMPLATE.format(
        question=question,
        reference_answer=reference_answer,
        context=context,
        answer=answer,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return _run_judge_with_retry(messages, "answerable")

def judge_refusal(question: str, answer: str) -> dict[str, Any]:
    user_prompt = UNANSWERABLE_USER_TEMPLATE.format(
        question=question,
        answer=answer,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return _run_judge_with_retry(messages, "refusal")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["answerable", "refusal"], default="answerable")
    args = parser.parse_args()
    if args.mode == "answerable":
        res = judge_answerable("What is FastAPI?", "FastAPI is a Python web framework.", "[chunk-1] FastAPI is a Python web framework.", "FastAPI is a Python web framework.")
    else:
        res = judge_refusal("What is quantum gravity in FastAPI?", "Not found in the provided documents.")
    print(res)

if __name__ == "__main__":
    main()
