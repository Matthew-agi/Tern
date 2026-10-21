# Deployment validation

CPU tests check source integrity, profile equivalence, routing math, layer
selection, head equations, adapter merging, lifecycle failures, download integrity
and benchmark accounting. They do not execute CUDA. On a CPU
development machine, install the CPU test environment:

```bash
uv venv --python 3.12
uv pip install --no-deps -e .
uv pip install pytest httpx pyyaml build safetensors==0.8.0 numpy==2.4.6
uv pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pytest -q
```

## On the GPU host

1. Build and launch the chosen profile; save logs with the GPU name/driver,
   revision and exact command. Confirm `/ready` and inspect all probability
   fields in the synthetic example. A source-contract or unsupported-kernel
   error must be investigated, not bypassed with eager mode or wider tolerances.
2. Use a fixed local JSONL panel of representative SystemOne request bodies,
   one per line. Include multiple question types, multiple decisions per record
   and natural lengths up to the supported limit. Keep benchmark inputs private.
3. Run the complete panel once at the intended concurrency before collecting
   repeated measurements, so relevant shapes are warmed. `--warmup` alone repeats
   only the first batch and does not warm every possible shape.
4. Repeat serial measurements, then the same panel at concurrency 128 for A100
   or 256 for H100, then repeat serial. Compare identical payloads. Report mean,
   p50/p80/p95/p99 latency, length buckets, throughput and any probability/argmax
   drift separately. Existing cross-batch variation is disclosed, not a pass.
5. On H100, collect BF16 reference results before the FP8 profile. Evaluate
   quality with the official scorer on authorized held-out inputs; output
   comparison alone cannot determine accuracy or Brier score.

Native CLI examples (inside the installed Python environment):

```bash
tern benchmark --input panel.jsonl --output measurements/serial-1 --concurrency 1
tern benchmark --input panel.jsonl --output measurements/serial-2 --concurrency 1
tern compare measurements/serial-1 measurements/serial-2
tern benchmark --input panel.jsonl --output measurements/a100-load --concurrency 128
tern compare measurements/serial-1 measurements/a100-load
tern benchmark --input panel.jsonl --output measurements/http-batches --concurrency 8 --batch-size 16
```

For Docker, create a host output directory, then copy the local panel into the
running container and invoke the same packaged CLI:

```bash
mkdir -p measurements
docker compose cp panel.jsonl tern:/tmp/panel.jsonl
docker compose exec tern tern benchmark --input /tmp/panel.jsonl --output /tmp/serial-1 --concurrency 1
docker compose cp tern:/tmp/serial-1 measurements/serial-1
```

Use a fresh output path for every run. Failed responses make the run incomplete
and suppress the headline throughput. `responses.jsonl` stores payload hashes,
latencies, server timing and full model responses; `summary.json` stores aggregate
metrics. Batch latency is whole-batch completion time per record, not batch time
divided by its size. Concurrent latency includes server queueing but not waiting
in a client dispatch queue. The synthetic example is not a representative panel.

Throughput is total successful **encoded input tokens** divided by warmed wall
time. This includes schema tokens and excludes load, compilation and idle time.
GPU-only cost per million input tokens is `hourly_rate * 1e6 / (3600 * tok_s)`.
Length buckets have value only when there are enough natural inputs in each bin.

For sustained testing, provide enough distinct representative inputs for several
minutes, record CPU/GPU utilization alongside it, and keep profiling separate
from the throughput run. Do not stack historical isolated gains. The repo has
no automatic quality acceptance threshold and does not turn comparison output
into a benchmark submission.

## Current status

- A100 frozen source and settings originate from the completed final evaluation.
- H100 fast settings now originate from the final October 20-21 campaign; see [scope and results](execution-policies.md).
- The new Docker image, RTX PRO 6000 startup, HTTP contract and warmed latency/throughput were tested on 2026-10-18. See [results and numerical limits](rtx-validation.md). Cross-GPU and cross-batch probability equivalence failed; serial repeats were exact.
- This branch has CPU tests and historical GPU source provenance, but no new packaged GPU validation. The maintainers' private RTX acceptance is separate.

For RTX restart checks, compare fresh processes, not only repeated requests in one process. The [workspace investigation](numerical-reproducibility.md) records the fixed execution setting, rollback control, head hash checks and labeled smoke scope.

The [complete RTX public evaluation](decision-index-rtx.md) includes independent rescoring and a post-evaluation serial repeatability check. The Docker launcher applies the tested workspace setting without a manual environment step.

## Validating the new execution policies

Use the same checkpoint and execution policy on both sides when validating this
packaging against a saved campaign. A top-4 versus top-8 or depth-30 versus
depth-40 comparison measures a deliberate model change, not implementation
parity. Do not widen a numerical gate because those comparisons differ.

For each intended policy, start with `tern plan`, then check the startup log and
`/ready`. Repeat a fixed serial panel across fresh processes, warm the full load
panel, and measure three repetitions at the recorded concurrency. Test multi-field
requests and mixed input lengths. Keep profiles and traces out of timed runs.
Compare to saved reference predictions/tokens when available; rebuilding old
baselines is not a prerequisite for preparing or inspecting this package.

The new CUDA kernels and full 70GB continuation merge were not run on a GPU during
this branch's local validation. CPU tensor fixtures verify the export arithmetic
and key contract; the complete private-model merge and GPU serving remain an
integration check on the next GPU host.
