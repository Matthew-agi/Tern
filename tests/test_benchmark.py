import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from tern.benchmark import compare, run
from tern.validation import validate_response

REQUEST = {"state": "Synthetic invoice is paid", "questions": {"status": {
    "type": "choice", "instructions": "Invoice status?", "criteria": {"paid": "Paid", "unpaid": "Unpaid"}}}}


def answer():
    return {"answers": {"status": {"type": "choice", "choice": "paid", "confidence": .8,
            "probabilities": {"paid": .8, "unpaid": .2}}}, "usage": {"input_tokens": 100, "output_tokens": 0}}


@pytest.fixture
def server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            records = body["requests"] if self.path.endswith("/batch") else [body]
            with state["lock"]:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(.02)
            with state["lock"]:
                state["active"] -= 1
            failed = any(r["state"] == "FAIL" for r in records)
            self.send_response(500 if failed else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Server-Timing", "prepare;dur=1,vllm;dur=20,format;dur=1")
            self.end_headers()
            data = {"data": [answer() for _ in records]} if self.path.endswith("/batch") else answer()
            self.wfile.write(json.dumps(data).encode())
        def log_message(self, *args):
            pass
    state = {"active": 0, "peak": 0, "lock": threading.Lock()}
    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}", state
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


def arguments(tmp_path, server, batch_size=1):
    inputs = tmp_path / "input.jsonl"
    inputs.write_text("\n".join(json.dumps(REQUEST) for _ in range(9)) + "\n")
    return SimpleNamespace(input=inputs, output=tmp_path / "measurement", base_url=server[0],
                           concurrency=3, warmup=1, batch_size=batch_size)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_benchmark_counts_tokens_and_batch_latency(tmp_path, server, batch_size):
    args = arguments(tmp_path, server, batch_size)
    summary = asyncio.run(run(args))
    assert summary["complete"] and summary["successful"] == 9
    assert summary["input_tokens"] == 900
    assert 1 < server[1]["peak"] <= 3
    assert summary["input_tokens_per_second"] == 900 / summary["elapsed_seconds"]
    rows = [json.loads(line) for line in (args.output / "responses.jsonl").read_text().splitlines()]
    assert sorted(r["id"] for r in rows) == list(range(1, 10))
    assert all(r["latency_ms"] >= 20 and r["server_timing"] for r in rows)
    assert summary["latency_by_input_tokens_ms"]["0-511"]["count"] == 9
    assert compare(args.output, args.output)["maximum_probability_difference"] == 0
    with pytest.raises(ValueError, match="preserved"):
        asyncio.run(run(args))


def test_failed_requests_do_not_produce_headline_throughput(tmp_path, server):
    args = arguments(tmp_path, server)
    with args.input.open("a") as stream:
        stream.write(json.dumps({**REQUEST, "state": "FAIL"}) + "\n")
    summary = asyncio.run(run(args))
    assert summary["successful"] == 9 and summary["failed"] == 1
    assert not summary["complete"] and summary["input_tokens_per_second"] is None
    with pytest.raises(ValueError, match="failed"):
        compare(args.output, args.output)


def test_comparison_requires_paired_payloads_and_reports_drift(tmp_path):
    row = {"id": 1, "payload_sha256": "same", "response": answer()}
    ref, candidate = tmp_path / "ref.jsonl", tmp_path / "candidate.jsonl"
    ref.write_text(json.dumps(row) + "\n")
    row["response"]["answers"]["status"]["probabilities"] = {"paid": .1, "unpaid": .9}
    candidate.write_text(json.dumps(row) + "\n")
    result = compare(ref, candidate)
    assert result["argmax_changes"] == 1
    assert result["maximum_probability_difference"] == pytest.approx(.7)
    row["payload_sha256"] = "different"
    candidate.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="payload"):
        compare(ref, candidate)


def test_validation_checks_complete_option_space():
    response = answer()
    response["answers"]["status"]["probabilities"] = {"paid": 1.0}
    with pytest.raises(ValueError, match="incomplete"):
        validate_response(REQUEST, response)


def test_validation_accepts_score_and_noul_without_changing_them():
    request = {"questions": {"s": {"type": "score", "criteria": ["bad", "good"]}, "n": {"type": "noul"}}}
    response = {"answers": {"s": {"type": "score", "probabilities": {"0": .4, "1": .6}},
                            "n": {"type": "noul", "noul": .7}}, "usage": {"input_tokens": 1, "output_tokens": 0}}
    before = json.dumps(response)
    validate_response(request, response)
    assert json.dumps(response) == before
