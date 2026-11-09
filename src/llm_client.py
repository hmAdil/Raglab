import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
import requests

try:
    from config import (
        GEN_BASE_URL,
        GEN_MODEL,
        JUDGE_BASE_URL,
        JUDGE_API_KEY,
        JUDGE_MODEL,
        CACHE_DIR,
    )
except ImportError:
    ROOT = Path(__file__).resolve().parent.parent
    CACHE_DIR = ROOT / "cache"
    GEN_BASE_URL = os.environ.get("GEN_BASE_URL", "http://localhost:11434")
    GEN_MODEL = os.environ.get("GEN_MODEL", "llama3.2:3b")
    JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "")
    JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", "")
    JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "")

OLLAMA_TIMEOUT = 180
JUDGE_TIMEOUT = 60
BACKOFF_DELAYS = [2, 4, 8]
OLLAMA_NUM_CTX = 8192
OLLAMA_NUM_PREDICT = 400

def _get_cache_path(cache_payload: dict[str, Any]) -> Path:
    canonical = json.dumps(cache_payload, sort_keys=True, separators=(",", ":"))
    key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.json"

def _read_cache(cache_path: Path) -> dict[str, Any] | None:
    if not cache_path.is_file():
        return None
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    resp = data.get("response", {})
    resp["latency_ms"] = data.get("latency_ms", 0.0)
    return resp

def _write_cache(cache_path: Path, cache_payload: dict[str, Any], response_data: dict[str, Any], latency_ms: float) -> None:
    resp_copy = dict(response_data)
    resp_copy.pop("latency_ms", None)
    record = {
        "request": cache_payload,
        "response": resp_copy,
        "latency_ms": latency_ms,
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

def ollama_chat(prompt: str) -> dict[str, Any]:
    url = f"{GEN_BASE_URL.rstrip('/')}/api/chat"
    req_body = {
        "model": GEN_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": OLLAMA_NUM_PREDICT,
        },
    }
    cache_path = _get_cache_path(req_body)
    cached = _read_cache(cache_path)
    if cached is not None:
        return cached

    last_exc: Exception | None = None
    for attempt in range(len(BACKOFF_DELAYS) + 1):
        t0 = time.perf_counter()
        try:
            resp = requests.post(url, json=req_body, timeout=OLLAMA_TIMEOUT)
            if resp.status_code == 429 or resp.status_code >= 500:
                resp.raise_for_status()
            resp.raise_for_status()
            latency_ms = (time.perf_counter() - t0) * 1000.0
            data = resp.json()
            prompt_tokens = int(data.get("prompt_eval_count", 0))
            if prompt_tokens >= (OLLAMA_NUM_CTX - 8):
                raise RuntimeError(f"Ollama prompt truncated: prompt_tokens={prompt_tokens} >= num_ctx - 8")
            result = {
                "answer": data["message"]["content"],
                "prompt_tokens": prompt_tokens,
                "completion_tokens": int(data.get("eval_count", 0)),
                "latency_ms": latency_ms,
            }
            _write_cache(cache_path, req_body, result, latency_ms)
            return result
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as exc:
            last_exc = exc
            if isinstance(exc, requests.exceptions.HTTPError):
                status = exc.response.status_code if exc.response is not None else 0
                if status != 429 and status < 500:
                    raise exc
            if attempt < len(BACKOFF_DELAYS):
                time.sleep(BACKOFF_DELAYS[attempt])
            else:
                raise last_exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Ollama call failed without exception")

def openai_chat(messages: list[dict[str, Any]]) -> dict[str, Any]:
    url = f"{JUDGE_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {JUDGE_API_KEY}",
        "Content-Type": "application/json",
    }
    base_payload: dict[str, Any] = {
        "model": JUDGE_MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 300,
    }
    cache_payload = {
        "model": JUDGE_MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
    }
    cache_path = _get_cache_path(cache_payload)
    cached = _read_cache(cache_path)
    if cached is not None:
        return cached

    req_body = dict(base_payload)
    req_body["response_format"] = {"type": "json_object"}

    last_exc: Exception | None = None
    use_response_format = True

    for attempt in range(len(BACKOFF_DELAYS) + 1):
        t0 = time.perf_counter()
        try:
            resp = requests.post(url, headers=headers, json=req_body, timeout=JUDGE_TIMEOUT)
            if resp.status_code == 400 and use_response_format:
                use_response_format = False
                req_body.pop("response_format", None)
                resp = requests.post(url, headers=headers, json=req_body, timeout=JUDGE_TIMEOUT)
            if resp.status_code == 429 or resp.status_code >= 500:
                resp.raise_for_status()
            resp.raise_for_status()
            latency_ms = (time.perf_counter() - t0) * 1000.0
            data = resp.json()
            usage = data.get("usage", {})
            choice = data["choices"][0]
            result = {
                "content": choice["message"]["content"],
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "latency_ms": latency_ms,
            }
            _write_cache(cache_path, cache_payload, result, latency_ms)
            return result
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as exc:
            last_exc = exc
            if isinstance(exc, requests.exceptions.HTTPError):
                status = exc.response.status_code if exc.response is not None else 0
                if status != 429 and status < 500:
                    raise exc
            if attempt < len(BACKOFF_DELAYS):
                time.sleep(BACKOFF_DELAYS[attempt])
            else:
                raise last_exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Judge API call failed without exception")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-ollama", action="store_true")
    parser.add_argument("--prompt", default="What is FastAPI?")
    args = parser.parse_args()
    if args.test_ollama:
        res = ollama_chat(args.prompt)
        print(f"Ollama response received: {len(res['answer'])} characters, {res['latency_ms']:.1f} ms")

if __name__ == "__main__":
    main()
