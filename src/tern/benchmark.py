"""Bounded-concurrency measurement; inputs and responses stay on the test host."""
import asyncio
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import time

from .artifacts import sha256
from .validation import validate_response


def payload_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def records(path):
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                record = json.loads(line)
                if "state" not in record or not record.get("questions"):
                    raise ValueError(f"Expected a SystemOne request at line {line_number}")
                yield line_number, record


def percentiles(values):
    ordered = sorted(values)
    if not ordered:
        return {}
    def at(q):
        pos = (len(ordered) - 1) * q
        low = int(pos)
        return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (pos - low)
    return {"mean": statistics.mean(ordered), "p50": at(.5), "p80": at(.8),
            "p95": at(.95), "p99": at(.99), "max": ordered[-1]}


async def run(args):
    import httpx
    if args.concurrency < 1 or args.warmup < 0 or not 1 <= args.batch_size <= 256:
        raise ValueError("Use positive concurrency, nonnegative warmup, and batch-size 1..256")
    if args.output.exists():
        raise ValueError("Choose a new output directory; existing measurements are preserved")
    # Validate the entire stream before starting measurements.
    count = sum(1 for _ in records(args.input))
    if count == 0:
        raise ValueError("Input file has no requests")
    input_hash = sha256(args.input)
    args.output.mkdir(parents=True)
    endpoint = "/v1/systemone" if args.batch_size == 1 else "/v1/systemone/batch"
    latencies, record_latencies, token_latencies = [], [], []
    errors, input_tokens, completed = 0, 0, 0
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    async with httpx.AsyncClient(base_url=args.base_url, timeout=600,
            limits=httpx.Limits(max_connections=args.concurrency,
                               max_keepalive_connections=args.concurrency)) as client:
        response = await client.get("/ready")
        response.raise_for_status()
        async def send(batch):
            bodies = [body for _, body in batch]
            body = bodies[0] if args.batch_size == 1 else {"requests": bodies}
            result = await client.post(endpoint, json=body)
            result.raise_for_status()
            output = result.json()
            responses = [output] if args.batch_size == 1 else output["data"]
            if len(responses) != len(batch):
                raise ValueError("Wrong batch response length")
            for request, answer in zip(bodies, responses, strict=True):
                validate_response(request, answer)
            return responses, result.headers.get("Server-Timing")
        warm = list(itertools.islice(records(args.input), args.batch_size))
        for _ in range(args.warmup):
            await send(warm)
        source = records(args.input)
        with (args.output / "responses.jsonl").open("w") as output:
            async def worker():
                nonlocal errors, input_tokens, completed
                while batch := list(itertools.islice(source, args.batch_size)):
                    begin = time.perf_counter()
                    try:
                        answers, timing = await send(batch)
                        elapsed = (time.perf_counter() - begin) * 1000
                        latencies.append(elapsed)
                        for (ident, body), answer in zip(batch, answers, strict=True):
                            tokens = answer["usage"]["input_tokens"]
                            input_tokens += tokens
                            completed += 1
                            record_latencies.append(elapsed)
                            token_latencies.append((tokens, elapsed))
                            output.write(json.dumps({"id": ident, "payload_sha256": payload_hash(body),
                                "latency_ms": elapsed, "http_batch_size": len(batch),
                                "server_timing": timing, "response": answer}) + "\n")
                    except Exception as exc:
                        errors += len(batch)
                        for ident, body in batch:
                            output.write(json.dumps({"id": ident, "payload_sha256": payload_hash(body),
                                                     "error": str(exc)}) + "\n")
                    output.flush()
            begin = time.perf_counter()
            await asyncio.gather(*(worker() for _ in range(args.concurrency)))
            elapsed = time.perf_counter() - begin
    # An input changed during a run must not produce an apparently matched result.
    input_unchanged = sha256(args.input) == input_hash
    valid = errors == 0 and completed == count and input_unchanged
    result = {"complete": valid, "started_at": started_at, "input_sha256": input_hash,
        "input_unchanged": input_unchanged, "records": count, "successful": completed,
        "failed": errors, "concurrency": args.concurrency, "http_batch_size": args.batch_size,
        "warmup_http_requests": args.warmup, "elapsed_seconds": elapsed,
        "input_tokens": input_tokens,
        "input_tokens_per_second": input_tokens / elapsed if valid else None,
        "http_latency_ms": percentiles(latencies), "record_latency_ms": percentiles(record_latencies),
        "latency_by_input_tokens_ms": {},
        "scope": "Warmed client wall time, HTTP + encoding + inference. Batch latency is completion time for the whole batch, not divided by records. No queueing before client dispatch; no loading/compilation/idle cost."}
    for low, high in [(0, 512), (512, 2048), (2048, 8192), (8192, 16384), (16384, 32769)]:
        values = [ms for n, ms in token_latencies if low <= n < high]
        result["latency_by_input_tokens_ms"][f"{low}-{high-1}"] = {"count": len(values), **percentiles(values)}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def benchmark(args):
    result = asyncio.run(run(args))
    print(json.dumps(result, indent=2))
    if not result["complete"]:
        raise SystemExit("Measurement incomplete; no throughput headline reported")


def compare(reference, candidate):
    def load(path):
        path = path / "responses.jsonl" if path.is_dir() else path
        rows = {}
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["id"] in rows or "error" in row:
                raise ValueError("Cannot compare duplicate IDs or failed responses")
            rows[row["id"]] = row
        return rows
    left, right = load(reference), load(candidate)
    if not left or left.keys() != right.keys():
        raise ValueError("Comparison requires the same nonempty request IDs")
    deltas, total_variations, changes, decisions = [], [], 0, 0
    for ident, row in left.items():
        other = right[ident]
        if row["payload_sha256"] != other["payload_sha256"]:
            raise ValueError(f"Mismatched request payload for ID {ident}")
        a, b = row["response"]["answers"], other["response"]["answers"]
        if a.keys() != b.keys():
            raise ValueError("Mismatched decision keys")
        for key in a:
            if a[key]["type"] != b[key]["type"]:
                raise ValueError("Mismatched decision types")
            def probs(answer):
                if answer["type"] == "noul":
                    return {"true": answer["noul"], "false": 1 - answer["noul"]}
                return answer["probabilities"]
            pa, pb = probs(a[key]), probs(b[key])
            if not pa or pa.keys() != pb.keys():
                raise ValueError("Mismatched option keys")
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in [*pa.values(), *pb.values()]):
                raise ValueError("Invalid probabilities in comparison")
            diff = [abs(pa[k] - pb[k]) for k in pa]
            if any(not 0 <= d <= 1 for d in diff):
                raise ValueError("Invalid probabilities in comparison")
            deltas.extend(diff)
            total_variations.append(sum(diff) / 2)
            # Preserve the encoder's deterministic tie ordering for each type.
            kind = a[key]["type"]
            order = (list(pa) if kind == "noul" else sorted(pa, key=int) if kind == "score" else sorted(pa))
            changes += max(order, key=pa.get) != max(order, key=pb.get)
            decisions += 1
    return {"requests": len(left), "decisions": decisions, "argmax_changes": changes,
            "maximum_probability_difference": max(deltas), "mean_probability_difference": statistics.mean(deltas),
            "mean_total_variation": statistics.mean(total_variations),
            "scope": "Paired outputs, not ground-truth quality or a relaxed numerical acceptance gate."}
