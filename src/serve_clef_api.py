"""SystemOne API for the trained Clef MoE vLLM pooler."""
import asyncio
import json
import math
from contextlib import asynccontextmanager
import os
from pathlib import Path
import time

import httpx
from fastapi import FastAPI, HTTPException, Response
from transformers import AutoTokenizer
from kev_vllm.clef_schema import encode_record
from kev_vllm.clef_encoding import schema_metadata
from kev_vllm.tokenization import CachedTokenizer

ROOT = Path(os.environ.get("CLEF_ROOT", "/home/ubuntu/kev-clef-training/serving"))
MODEL = os.environ.get("CLEF_MODEL", "moe-clef-step350-runtime")
BACKEND = os.environ.get("CLEF_BACKEND", "http://127.0.0.1:8101")
model_directory = ROOT / MODEL
if (model_directory / "runtime.json").exists():
    MANIFEST = json.loads((model_directory / "runtime.json").read_text())
    NAME = MANIFEST["model_name"]
    MAX_LENGTH = MANIFEST["max_length"]
else:
    # A merged export is served as the base model, with no adapter identifier.
    MANIFEST = json.loads((model_directory / "export.json").read_text())
    NAME = "clef-base-35b"
    MAX_LENGTH = MANIFEST["max_length"]
MAX_LENGTH = int(os.environ.get("CLEF_MAX_LENGTH", MAX_LENGTH))
MAX_CONNECTIONS = int(os.environ.get("CLEF_MAX_CONNECTIONS", "128"))
tokenizer = CachedTokenizer(AutoTokenizer.from_pretrained(ROOT / MODEL, local_files_only=True))


@asynccontextmanager
async def lifespan(app):
    async with httpx.AsyncClient(base_url=BACKEND, timeout=600, limits=httpx.Limits(max_connections=MAX_CONNECTIONS, max_keepalive_connections=MAX_CONNECTIONS, keepalive_expiry=1)) as client:
        app.state.client = client
        app.state.pipeline = None
        mode = os.environ.get('CLEF_SUBMISSION_MODE', 'single')
        if mode != 'single':
            from kev_vllm.submission_pipeline import SubmissionPipeline
            app.state.pipeline = SubmissionPipeline(client, prepare_many, format_result, schema_metadata, NAME, mode=mode)
        try:
            yield
        finally:
            if app.state.pipeline:
                await app.state.pipeline.close()


app = FastAPI(lifespan=lifespan)


def prepare(request):
    if "state" not in request or not request.get("questions"):
        raise ValueError("state and questions are required")
    record = {"state": request["state"], "questions": {name: {key: q[key] for key in ("type", "instructions", "criteria") if key in q} for name, q in request["questions"].items()}}
    item = encode_record(tokenizer, record, max_length=1 << 30)
    if len(item.input_ids) > MAX_LENGTH:
        raise ValueError(f"Input exceeds the maximum model length of {MAX_LENGTH} tokens")
    return item


def prepare_many(requests):
    from kev_vllm.batch_encoding import encode_batch
    records = []
    for request in requests:
        if 'state' not in request or not request.get('questions'):
            raise ValueError('state and questions are required')
        records.append({'state': request['state'], 'questions': {
            name: {key: q[key] for key in ('type', 'instructions', 'criteria') if key in q}
            for name, q in request['questions'].items()}})
    return encode_batch(tokenizer, records, MAX_LENGTH)


def format_result(request, item, data):
    expected = sum(len(q.option_ids) for q in item.questions)
    if len(data) != expected:
        raise HTTPException(502, 'Invalid joint-head output shape')
    answers, offset = {}, 0
    for q in item.questions:
        n = len(q.option_ids)
        values = data[offset:offset+n]
        offset += n
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in values) or abs(sum(values)-1) > .001:
            raise HTTPException(502, 'Invalid joint-head probabilities')
        probabilities = dict(zip(q.option_ids, values, strict=True))
        confidence = max(values)
        if q.question_type == 0:
            answer = {'type': 'noul', 'noul': probabilities['true']}
        elif q.question_type == 1:
            answer = {'type': 'choice', 'choice': max(probabilities, key=probabilities.get),
                      'confidence': confidence, 'probabilities': probabilities}
        else:
            answer = {'type': 'score', 'score': sum(int(k)*v for k, v in probabilities.items()),
                      'confidence': confidence, 'legend': request['questions'][q.question_id]['criteria'],
                      'probabilities': probabilities}
        answers[q.question_id] = answer
    return {'model': NAME, 'answers': answers, 'usage': {'input_tokens': len(item.input_ids), 'output_tokens': 0}}


@app.post('/v1/systemone/batch')
async def systemone_batch(request: dict, http_response: Response):
    requests = request.get('requests')
    if not isinstance(requests, list) or not 1 <= len(requests) <= 256:
        raise HTTPException(422, 'Expected 1 to 256 complete records')
    started = time.perf_counter()
    try:
        items = await asyncio.to_thread(prepare_many, requests)
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    prepared = time.perf_counter()
    response = await app.state.client.post('/pooling', json={'model': NAME,
        'input': [list(item.input_ids) for item in items], 'task': 'classify', 'encoding_format': 'float',
        'clef_records': [schema_metadata(item) for item in items]})
    if response.status_code != 200:
        raise HTTPException(502, response.text)
    data = sorted(response.json()['data'], key=lambda row: row['index'])
    if [row['index'] for row in data] != list(range(len(items))):
        raise HTTPException(502, 'Incomplete or reordered batch response')
    scored = time.perf_counter()
    outputs = [format_result(record, item, row['data']) for record, item, row in zip(requests, items, data, strict=True)]
    http_response.headers['Server-Timing'] = f'prepare;dur={(prepared-started)*1000:.3f},vllm;dur={(scored-prepared)*1000:.3f},format;dur={(time.perf_counter()-scored)*1000:.3f}'
    return {'data': outputs}


@app.get("/health")
async def health():
    response = await app.state.client.get("/health")
    if response.status_code != 200:
        raise HTTPException(503, "vLLM backend unavailable")
    return {"status": "ok", "model": NAME, "backend": "vllm-clef-joint-pooler"}


@app.post("/v1/systemone")
async def systemone(request: dict, http_response: Response):
    if app.state.pipeline:
        try:
            result, timing = await app.state.pipeline.submit(request)
            http_response.headers['Server-Timing'] = timing
            return result
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            raise HTTPException(502, exc.response.text) from exc
    t = time.perf_counter()
    try:
        item = await asyncio.to_thread(prepare, request)
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    prepared = time.perf_counter()
    response = await app.state.client.post("/pooling", json={"model": NAME, "input": [list(item.input_ids)],
        "task": "classify", "encoding_format": "float", "clef_record": schema_metadata(item)})
    if response.status_code != 200:
        raise HTTPException(502, response.text)
    result = response.json()
    data = result["data"][0]["data"]
    scored = time.perf_counter()
    expected = sum(len(q.option_ids) for q in item.questions)
    if len(data) != expected:
        raise HTTPException(502, "Invalid joint-head output shape")
    answers, offset = {}, 0
    for q in item.questions:
        n = len(q.option_ids)
        values = data[offset:offset+n]
        offset += n
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in values) or abs(sum(values) - 1) > .001:
            raise HTTPException(502, "Invalid joint-head probabilities")
        probabilities = dict(zip(q.option_ids, values, strict=True))
        confidence = max(values)
        if q.question_type == 0:
            answer = {"type": "noul", "noul": probabilities["true"]}
        elif q.question_type == 1:
            answer = {"type": "choice", "choice": max(probabilities, key=probabilities.get), "confidence": confidence, "probabilities": probabilities}
        else:
            answer = {"type": "score", "score": sum(int(k)*v for k,v in probabilities.items()), "confidence": confidence,
                "legend": request["questions"][q.question_id]["criteria"], "probabilities": probabilities}
        answers[q.question_id] = answer
    http_response.headers["Server-Timing"] = f"prepare;dur={(prepared-t)*1000:.3f},vllm;dur={(scored-prepared)*1000:.3f},format;dur={(time.perf_counter()-scored)*1000:.3f}"
    return {"model": NAME, "answers": answers, "usage": {"input_tokens": len(item.input_ids), "output_tokens": 0}}
