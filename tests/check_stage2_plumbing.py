import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any
from unittest.mock import patch, MagicMock

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

try:
    import config
except ImportError:
    config = None
import csv
import prompts
import llm_client
import generator
import judge
import oracle_retriever
import run_stage2
import judge_validation

APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
import gradio_app

def test_prompts() -> None:
    chunks = [
        {"chunk_id": "test-001", "text": "Chunk text 1"},
        {"chunk_id": "test-002", "text": "Chunk text 2"},
    ]
    ctx = prompts.format_context(chunks)
    expected_ctx = "[test-001] Chunk text 1\n\n[test-002] Chunk text 2"
    assert ctx == expected_ctx, f"Context mismatch: {repr(ctx)} vs {repr(expected_ctx)}"

    q = "What is RAG?"
    plain = prompts.build_prompt("plain", q, ctx)
    expected_plain = f"Answer the question using the context.\n\nContext:\n{ctx}\n\nQuestion: {q}\n\nAnswer:"
    assert plain == expected_plain, f"Plain prompt mismatch: {repr(plain)}"

    grounded = prompts.build_prompt("grounded", q, ctx)
    expected_grounded = f"Answer the question using only the context below. Cite the chunk IDs you used in square brackets, for example [fixed-000123]. If the context does not contain the answer, reply exactly: Not found in the provided documents.\n\nContext:\n{ctx}\n\nQuestion: {q}\n\nAnswer:"
    assert grounded == expected_grounded, f"Grounded prompt mismatch: {repr(grounded)}"

    try:
        prompts.build_prompt("invalid_style", q, ctx)
        assert False, "Should have raised ValueError on invalid style"
    except ValueError:
        pass

def test_llm_client() -> None:
    temp_dir = Path(tempfile.mkdtemp())
    try:
        with patch.object(llm_client, "CACHE_DIR", temp_dir):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "message": {"content": "Sample Ollama response"},
                "prompt_eval_count": 25,
                "eval_count": 15,
            }

            with patch("requests.post", return_value=mock_resp) as mock_post:
                res1 = llm_client.ollama_chat("Hello Ollama")
                assert res1["answer"] == "Sample Ollama response"
                assert res1["prompt_tokens"] == 25
                assert res1["completion_tokens"] == 15
                assert "latency_ms" in res1
                assert mock_post.call_count == 1
                posted_json = mock_post.call_args[1]["json"]
                assert posted_json["model"] == llm_client.GEN_MODEL
                assert posted_json["options"]["temperature"] == 0
                assert posted_json["options"]["seed"] == 42
                assert posted_json["options"]["num_ctx"] == 8192
                assert posted_json["options"]["num_predict"] == 400

                res2 = llm_client.ollama_chat("Hello Ollama")
                assert mock_post.call_count == 1, "Second identical call must be served from cache"
                assert res2["answer"] == res1["answer"]
                assert res2["latency_ms"] == res1["latency_ms"]

            cache_files = list(temp_dir.glob("*.json"))
            assert len(cache_files) == 1
            with open(cache_files[0], "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            assert "Authorization" not in json.dumps(cached_data)

            mock_trunc = MagicMock()
            mock_trunc.status_code = 200
            mock_trunc.json.return_value = {
                "message": {"content": "Truncated"},
                "prompt_eval_count": 8185,
                "eval_count": 5,
            }
            with patch("requests.post", return_value=mock_trunc):
                try:
                    llm_client.ollama_chat("Trigger truncation error")
                    assert False, "Expected RuntimeError on prompt truncation"
                except RuntimeError:
                    pass

            mock_judge_resp = MagicMock()
            mock_judge_resp.status_code = 200
            mock_judge_resp.json.return_value = {
                "choices": [{"message": {"content": '{"groundedness": 5, "correctness": 5, "justification": "Good."}'}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 20},
            }
            with patch("requests.post", return_value=mock_judge_resp) as mock_judge_post:
                jres = llm_client.openai_chat([{"role": "user", "content": "Evaluate"}])
                assert "choices" not in jres
                assert jres["content"] == '{"groundedness": 5, "correctness": 5, "justification": "Good."}'
                assert jres["prompt_tokens"] == 50
                assert jres["completion_tokens"] == 20
                assert mock_judge_post.call_count == 1
                headers = mock_judge_post.call_args[1]["headers"]
                assert "Authorization" in headers

            attempts = 0
            def fail_then_succeed(*args: Any, **kwargs: Any) -> MagicMock:
                nonlocal attempts
                attempts += 1
                if attempts < 3:
                    resp_err = MagicMock()
                    resp_err.status_code = 500
                    resp_err.raise_for_status.side_effect = llm_client.requests.exceptions.HTTPError(response=resp_err)
                    return resp_err
                ok_resp = MagicMock()
                ok_resp.status_code = 200
                ok_resp.json.return_value = {
                    "message": {"content": "Succeeded after retry"},
                    "prompt_eval_count": 10,
                    "eval_count": 10,
                }
                return ok_resp

            with patch("requests.post", side_effect=fail_then_succeed), patch("time.sleep", return_value=None):
                retry_res = llm_client.ollama_chat("Test retry flow")
                assert retry_res["answer"] == "Succeeded after retry"
                assert attempts == 3
    finally:
        shutil.rmtree(temp_dir)

def test_generator() -> None:
    chunks = [{"chunk_id": "gen-01", "text": "Python documentation."}]
    fake_chat = {
        "answer": "Generated answer.",
        "prompt_tokens": 30,
        "completion_tokens": 12,
        "latency_ms": 150.0,
    }
    with patch("generator.ollama_chat", return_value=fake_chat):
        res = generator.generate("plain", "Question?", chunks)
        assert res["answer"] == "Generated answer."
        assert res["prompt_tokens"] == 30
        assert res["completion_tokens"] == 12
        assert res["generation_ms"] == 150.0

def test_judge() -> None:
    raw_answerable = """```json
{"groundedness": 5, "correctness": 4, "justification": "Accurate with minor omission."}
```"""
    parsed = judge._parse_judge_json(raw_answerable, "answerable")
    assert parsed["groundedness"] == 5
    assert parsed["correctness"] == 4
    assert parsed["justification"] == "Accurate with minor omission."

    raw_refusal = '{"refused": true, "justification": "Properly refused."}'
    parsed_ref = judge._parse_judge_json(raw_refusal, "refusal")
    assert parsed_ref["refused"] is True
    assert parsed_ref["justification"] == "Properly refused."

    try:
        judge._parse_judge_json('{"groundedness": true, "correctness": 5, "justification": "Invalid"}', "answerable")
        assert False, "Boolean must not be accepted as integer score"
    except ValueError:
        pass

    try:
        judge._parse_judge_json('{"groundedness": 6, "correctness": 5, "justification": "Out of range"}', "answerable")
        assert False, "Score > 5 must be rejected"
    except ValueError:
        pass

    try:
        judge._parse_judge_json('{"refused": "yes", "justification": "Not a bool"}', "refusal")
        assert False, "Non-boolean refused must be rejected"
    except ValueError:
        pass

    call_count = 0
    def bad_then_good_chat(messages: list[dict[str, Any]]) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"content": "Not JSON at all", "prompt_tokens": 10, "completion_tokens": 10, "latency_ms": 10.0}
        return {
            "content": '{"groundedness": 4, "correctness": 4, "justification": "Parsed after retry."}',
            "prompt_tokens": 15,
            "completion_tokens": 15,
            "latency_ms": 15.0,
        }

    with patch.object(judge, "JUDGE_BASE_URL", "https://api.openai.com/v1"), patch.object(judge, "JUDGE_MODEL", "gpt-4o"):
        with patch("judge.openai_chat", side_effect=bad_then_good_chat):
            res = judge.judge_answerable("Q", "Ref", "Ctx", "Ans")
            assert call_count == 2
            assert res["groundedness"] == 4
            assert res["judge_error"] is None

        with patch("judge.openai_chat", return_value={"content": "Never JSON", "prompt_tokens": 10, "completion_tokens": 10, "latency_ms": 10.0}):
            res_fail = judge.judge_answerable("Q", "Ref", "Ctx", "Ans")
            assert res_fail["judge_error"] is not None
            assert res_fail["groundedness"] is None

def test_oracle_retriever() -> None:
    retriever = oracle_retriever.OracleRetriever(seed=42)
    assert retriever.config_id == "oracle"

    ans_q = {
        "id": "test_q1",
        "question": "What is the purpose of response_model in FastAPI?",
        "reference_answer": "It defines response validation.",
        "gold_evidence": {
            "source": "fastapi",
            "section": "response_model Parameter \u00b6",
            "span": "FastAPI will use this response_model to do all the data documentation, validation",
        },
        "answerable": True,
    }
    hits = retriever.retrieve_for_question(ans_q, k=5)
    assert len(hits) == 5
    assert hits[0].rank == 1
    assert hits[0].score == 1.0
    rec0 = retriever.get_chunk_record(hits[0].chunk_id)
    assert "FastAPI will use this response_model" in rec0["text"]
    assert hits[0].chunk_id.startswith("oracle-fastapi-")

    unans_q = {
        "id": "test_q2",
        "question": "How do I build quantum circuits in FastAPI?",
        "reference_answer": "",
        "answerable": False,
    }
    unans_hits = retriever.retrieve_for_question(unans_q, k=5)
    assert len(unans_hits) == 5
    assert unans_hits[0].rank == 1

    retriever2 = oracle_retriever.OracleRetriever(seed=42)
    hits2 = retriever2.retrieve_for_question(ans_q, k=5)
    assert [h.chunk_id for h in hits] == [h.chunk_id for h in hits2], "Retriever must be deterministic with SEED"

def test_run_stage2_plumbing() -> None:
    temp_dir = Path(tempfile.mkdtemp())
    try:
        runs_path = temp_dir / "stage2_runs.jsonl"
        csv_path = temp_dir / "stage2.csv"
        meta_path = temp_dir / "stage2_meta.json"

        sample_questions = [
            {
                "id": "q1",
                "question": "What is response_model?",
                "reference_answer": "Data validation and conversion.",
                "gold_evidence": {
                    "source": "fastapi",
                    "section": "response_model Parameter \u00b6",
                    "span": "FastAPI will use this response_model",
                },
                "answerable": True,
            },
            {
                "id": "q2",
                "question": "Quantum computing in FastAPI?",
                "reference_answer": "",
                "answerable": False,
            },
        ]

        def fake_generate(style: str, question: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
            return {
                "answer": f"Answer for {question} ({style})",
                "prompt_tokens": 40,
                "completion_tokens": 10,
                "generation_ms": 120.0,
            }

        def fake_judge_answerable(q: str, ref: str, ctx: str, ans: str) -> dict[str, Any]:
            return {"groundedness": 5, "correctness": 4, "justification": "Accurate.", "judge_error": None}

        def fake_judge_refusal(q: str, ans: str) -> dict[str, Any]:
            return {"refused": True, "justification": "Refused properly.", "judge_error": None}

        with patch.object(run_stage2, "RESULTS_DIR", temp_dir), \
             patch("run_stage2.load_testset", return_value=sample_questions), \
             patch("run_stage2.generate", side_effect=fake_generate), \
             patch("run_stage2.judge_answerable", side_effect=fake_judge_answerable), \
             patch("run_stage2.judge_refusal", side_effect=fake_judge_refusal):

            run_stage2.run_stage2(["oracle"], ["plain", "grounded"], limit=2, resume=False)

            assert runs_path.is_file()
            assert csv_path.is_file()
            assert meta_path.is_file()

            records = []
            with open(runs_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            assert len(records) == 4

            expected_keys = {
                "run_id", "config_id", "prompt_style", "question_id", "answerable",
                "answer", "retrieved_chunk_ids", "retrieval_ms", "generation_ms",
                "prompt_tokens", "completion_tokens", "judge", "judge_error"
            }
            assert set(records[0].keys()) == expected_keys

            with open(csv_path, "r", encoding="utf-8") as f:
                csv_lines = [l.strip() for l in f if l.strip()]
            header = csv_lines[0].split(",")
            expected_header = [
                "run_id", "config_id", "prompt_style", "n_answerable", "n_unanswerable",
                "groundedness", "correctness", "refusal_accuracy", "retrieval_latency_ms",
                "generation_latency_ms", "prompt_tokens", "completion_tokens"
            ]
            assert header == expected_header
            assert len(csv_lines) == 3

            call_count = 0
            def counting_generate(*args: Any, **kwargs: Any) -> dict[str, Any]:
                nonlocal call_count
                call_count += 1
                return fake_generate(*args, **kwargs)

            with patch("run_stage2.generate", side_effect=counting_generate):
                run_stage2.run_stage2(["oracle"], ["plain", "grounded"], limit=2, resume=True)
                assert call_count == 0, "Resume must skip completed (run_id, question_id) entries"

    finally:
        shutil.rmtree(temp_dir)

def test_single_oracle_end_to_end() -> None:
    retriever = oracle_retriever.OracleRetriever(seed=42)
    sample_q = {
        "id": "e2e_q1",
        "question": "What is response_model?",
        "reference_answer": "Validates data.",
        "gold_evidence": {
            "source": "fastapi",
            "section": "response_model Parameter \u00b6",
            "span": "FastAPI will use this response_model",
        },
        "answerable": True,
    }
    hits = retriever.retrieve_for_question(sample_q, k=5)
    chunks = [retriever.get_chunk_record(h.chunk_id) for h in hits]
    ctx = prompts.format_context(chunks)
    prompt = prompts.build_prompt("plain", sample_q["question"], ctx)
    assert len(prompt) > 0

    with patch("generator.ollama_chat", return_value={"answer": "Validation model", "prompt_tokens": 50, "completion_tokens": 10, "latency_ms": 100.0}):
        gen_res = generator.generate("plain", sample_q["question"], chunks)
        assert gen_res["answer"] == "Validation model"

    with patch.object(judge, "JUDGE_BASE_URL", "https://api.openai.com/v1"), patch.object(judge, "JUDGE_MODEL", "gpt-4o"):
        with patch("judge.openai_chat", return_value={"content": '{"groundedness": 5, "correctness": 5, "justification": "Perfect match."}', "prompt_tokens": 60, "completion_tokens": 15, "latency_ms": 50.0}):
            j_res = judge.judge_answerable(sample_q["question"], sample_q["reference_answer"], ctx, gen_res["answer"])
            assert j_res["groundedness"] == 5
            assert j_res["correctness"] == 5
            assert j_res["judge_error"] is None

def test_judge_validation() -> None:
    temp_dir = Path(tempfile.mkdtemp())
    try:
        runs_path = temp_dir / "stage2_runs.jsonl"
        sheet_path = temp_dir / "judge_validation_sheet.csv"
        out_json_path = temp_dir / "judge_validation.json"

        records: list[dict[str, Any]] = []
        for i in range(12):
            records.append({
                "run_id": f"run_0{i%3}",
                "config_id": "oracle",
                "prompt_style": "plain",
                "question_id": f"q_{i}",
                "answerable": True,
                "answer": f"Answer text {i}",
                "retrieved_chunk_ids": [f"oracle-fastapi-{i}"],
                "retrieval_ms": 10.0,
                "generation_ms": 100.0,
                "prompt_tokens": 50,
                "completion_tokens": 20,
                "judge": {
                    "groundedness": 5 if i % 2 == 0 else 4,
                    "correctness": 4 if i % 2 == 0 else 3,
                    "justification": f"Valid score for {i}",
                },
                "judge_error": None,
            })
        records.append({
            "run_id": "run_err",
            "config_id": "oracle",
            "prompt_style": "plain",
            "question_id": "q_err",
            "answerable": True,
            "answer": "Err",
            "retrieved_chunk_ids": ["oracle-fastapi-0"],
            "retrieval_ms": 10.0,
            "generation_ms": 10.0,
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "judge": None,
            "judge_error": "Failed",
        })

        with open(runs_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        mock_testset = [{"id": f"q_{i}", "question": f"Q {i}?", "reference_answer": f"Ref {i}"} for i in range(12)]

        with patch.object(judge_validation, "RESULTS_DIR", temp_dir), \
             patch("judge_validation.load_testset", return_value=mock_testset):

            judge_validation.make_validation_sheet()
            assert sheet_path.is_file()

            sheet_rows = []
            with open(sheet_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    sheet_rows.append(r)

            assert len(sheet_rows) == 10
            for r in sheet_rows:
                assert r["manual_groundedness"] == ""
                assert r["manual_correctness"] == ""
                assert "5" not in r["manual_groundedness"]
                assert "4" not in r["manual_correctness"]

            filled_rows = []
            for r in sheet_rows:
                row_copy = dict(r)
                row_copy["manual_groundedness"] = "5"
                row_copy["manual_correctness"] = "4"
                filled_rows.append(row_copy)

            with open(sheet_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(sheet_rows[0].keys()))
                writer.writeheader()
                writer.writerows(filled_rows)

            judge_validation.generate_validation_report()
            assert out_json_path.is_file()

            with open(out_json_path, "r", encoding="utf-8") as f:
                report_data = json.load(f)

            assert report_data["n_samples"] == 10
            assert "groundedness" in report_data
            assert "correctness" in report_data
            assert "exact_agreement" in report_data["groundedness"]
            assert "within_one_agreement" in report_data["groundedness"]
            assert "mean_abs_diff" in report_data["groundedness"]

    finally:
        shutil.rmtree(temp_dir)

def test_gradio_app() -> None:
    app = gradio_app.build_app()
    assert isinstance(app, gradio_app.gr.Blocks)

    gradio_app.RETRIEVER_CACHE.clear()
    gradio_app.COLBERT_CACHE_KEYS.clear()

    mock_ret = MagicMock()
    with patch.object(gradio_app, "get_retriever", return_value=mock_ret):
        gradio_app._get_or_create_retriever("fixed", "bge", "colbert")
        gradio_app._get_or_create_retriever("sentence", "bge", "colbert")
        assert len(gradio_app.COLBERT_CACHE_KEYS) == 2

        gradio_app._get_or_create_retriever("semantic", "bge", "colbert")
        assert len(gradio_app.COLBERT_CACHE_KEYS) == 2
        assert ("fixed", "bge", "colbert") not in gradio_app.RETRIEVER_CACHE
        assert ("semantic", "bge", "colbert") in gradio_app.RETRIEVER_CACHE

    fake_retriever = MagicMock()
    fake_retriever.retrieve.return_value = []
    with patch.object(gradio_app, "_get_or_create_retriever", return_value=fake_retriever), \
         patch("gradio_app.generate", side_effect=RuntimeError("Ollama connection error")):
        rows, ans, metrics = gradio_app.handle_playground_submit("fixed", "bge", "colbert", "plain", "Test question")
        assert "Ollama is unavailable" in ans
        assert "Generation failed" in metrics

def main() -> None:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    try:
        test_prompts()
        test_llm_client()
        test_generator()
        test_judge()
        test_oracle_retriever()
        test_run_stage2_plumbing()
        test_single_oracle_end_to_end()
        test_judge_validation()
        test_gradio_app()
        print("PASS")
        sys.exit(0)
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
