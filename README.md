# Tern

Typed decisions with probabilities, served by the trained joint head on a
Qwen3.6-35B-A3B backbone. One request supplies a state and named questions with
their complete answer options. Tern returns every decision in one forward pass;
it does not generate text.

This repo defaults to the final **step-1674, 40-layer, top-8** model. It also
supports explicit expert-count pruning, unconditional early exit and the pinned
trained 4/8 continuation. The original head definition, encoding and scoring API
are preserved; optimized execution sources are versioned and checked in CI.
Weights and the trained head live together on [Hugging Face](https://huggingface.co/matthewagi/Tern-35B-A3B),
pinned by revision and SHA-256. Code is Apache-2.0; the owner has selected
Apache-2.0 for Tern. Original dependency notices and [training disclosures](docs/training-disclosure.md)
are retained. This repo does not change Hub visibility or submit benchmark results.

**Tested on RTX PRO 6000 Blackwell 96GB:** a fresh Docker build, GPU kernel checks,
HTTP protocol, and warmed latency/throughput completed. [Numerical limits](docs/rtx-validation.md)
remain: serial repeats were exact, but cross-GPU and cross-batch outputs differed.
The default Docker launcher includes the cuBLAS workspace fix automatically. The A100 profile preserves its recorded settings. The updated H100 profile integrates the final campaign kernels; this branch still needs packaged GPU validation.

## Start on RTX PRO 6000

Use a Linux x86_64 host with the 96GB Blackwell card, a CUDA-12.9-compatible NVIDIA
driver, Docker Compose v2 and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Plan for at least **120GB free disk** for the 69.8GB model, image and compilation
caches; **128GB host RAM** is a provisioning recommendation, not a measured minimum.
The 48GB RTX 6000 Ada is a different card and is not supported by this BF16 profile.

```bash
git clone https://github.com/Matthew-agi/Tern.git
cd Tern
docker compose up --build
```

The code and model are public; no access token is required. On first
launch, Tern downloads only the 33 pinned inference files, verifies every hash,
loads the model, compiles kernels and runs a synthetic inference. Wait for
`Tern READY` or a healthy container before sending work. Persistent caches are
reused on later launches. A changed or incomplete local model is rejected.

```bash
curl -fsS http://localhost:8000/ready
curl -fsS http://localhost:8000/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @examples/request.json
```

The public port binds to localhost; use an SSH tunnel for a remote personal test:
`ssh -L 8000:127.0.0.1:8000 user@gpu-host`. `docker compose down` stops the service
without deleting the model cache. Do not add `--volumes` unless you intend to
delete that cache.

## Select a GPU profile

Stop the current service before switching. `TERN_GPU` selects one host GPU.

```bash
TERN_PROFILE=a100 TERN_GPU=0 docker compose up --build
TERN_PROFILE=h100 TERN_GPU=0 docker compose up --build
TERN_PROFILE=h100-bf16 TERN_GPU=0 docker compose up --build
```

| Profile | Execution | Recorded input throughput | Evidence boundary |
|---|---|---:|---|
| `rtx-pro-6000` (default) | BF16, Triton recurrent kernels, original head batching | 34,787 tok/s mean of 3 warmed bursts | Final step 1674; [deployment test and numerical differences](docs/rtx-validation.md), not private Index acceptance |
| `a100` | BF16, A100 tuned MoE tiles | 24,514 tok/s matched panel | A100 SXM4 80GB, final step 1674; measured before this repackaging |
| `h100` | FP8, FlashInfer, tuned MoE, fused preparation/quantization, compiled and vectorized head | 56,259 tok/s original 8/8; 65,509 trained 4/8; 84,475 trained 4/8 at layer 30 | H100 PCIe 80GB, final campaign; historical measurements before this packaging |
| `h100-bf16` | BF16 reference with original head batching | Unmeasured | Control for checking the H100 FP8 profile on final weights |

All use 32,768 context tokens, 256 sequences, 16,384-token piecewise CUDA
graphs and stateless complete-prefill execution. H100 uses a 65,536-token batch budget;
other profiles use 32,768. The known winning client loads
are 128 concurrent requests on A100 and 256 on H100. Client concurrency is not a
server flag. See [exact settings and conversions](docs/deployment.md) and
[recorded evidence](docs/profile-evidence.json).

The H100 profile preserves the learned operations but FP8, fusion and changed
execution grouping can change probabilities. Expert pruning and early exit are
explicit model choices with their own quality tradeoffs. Default RTX/A100
execution retains the BF16 path.

## Choose expert count and exit depth

```bash
TERN_PROFILE=h100 TERN_EXPERTS=4 docker compose up --build
TERN_PROFILE=h100 TERN_EXPERTS=6 TERN_EXIT_LAYER=30 docker compose up --build
```

Native equivalents are `tern serve --profile h100 --experts 4` and
`tern serve --profile h100 --experts 6 --exit-layer 30`. Routing retains top-8
normalization by default. Exit is unconditional and keeps the original final norm
and head. These choices also work with the A100/RTX profiles, with GPU validation
required for new combinations.

For the **trained** 4/8 rows, add `--adapter top4-50m` (or `TERN_ADAPTER=top4-50m`)
with access to the private HF adapter. This automatically merges the pinned LoRA
and trained head, and defaults to four experts. Pruning alone uses original
weights. See [execution policies, commands and measurement scope](docs/execution-policies.md).

## Native installation and testing

Docker is the recommended deployment path. For native Linux Python 3.12, install
the exact [dependency lock](requirements.lock), then this package:

```bash
uv venv --python 3.12
uv pip sync requirements.lock \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match
uv pip install --no-deps .
.venv/bin/tern check-runtime
.venv/bin/tern serve --profile rtx-pro-6000
```

The native host also needs CUDA 12.9's development toolkit/compiler and compatible
driver. `tern serve --model /path/to/snapshot` uses an existing verified copy.
`tern plan --profile h100` prints the complete vLLM command without GPU work.
vLLM's source contract is checked before a model download or launch; upgrading
vLLM independently is unsupported.

The [test guide](docs/testing.md) covers serial/concurrent latency, explicit
HTTP batches, length buckets and paired probability comparison. The benchmark
client records responses locally and never uploads inputs. The small synthetic
example proves API wiring; it is not a quality or performance benchmark.

## Model and evaluation

The [complete RTX public evaluation](docs/decision-index-rtx.md) scores **59.01 public / 68.93 raw**, with 140,176 successful requests, two declared capacity rejections and zero errors. Independent rescoring reproduced every score field. The original A100 result remains **59.10 / 69.04**. See the [model card](docs/model-card.md) for both runs.
A separate corrected SGD auxiliary check scores **0.7033 macro-F1**; the frozen suite had an upstream dialogue-history bug. See the [disclosure](docs/decision-index-rtx.md#upstream-sgd-auxiliary-input-correction); the headline index is unaffected.
This is not an accepted leaderboard placement or the private Full score.
Cross-batch numerical variation and training/development overlap are disclosed;
we do not promise identical probabilities across batches or GPU types.

Directory layout:

```text
src/tern/          CLI, lifecycle, profiles, artifact verification, benchmark client
src/kev_vllm/      Model, joint head, encoding, source-checked kernels and vLLM plugin
src/serve_clef_api.py  Frozen SystemOne API; Tern wrapper changes only service identity
examples/          Synthetic request (no benchmark inputs)
tests/             CPU contracts and packaging checks
docs/              Architecture, GPU testing, provenance and disclosures
```

See [architecture](docs/architecture.md) for how the backbone, trained head and
API connect. Large weights remain on HF; code, launch profiles and compact
provenance belong here. Training data, benchmark inputs, optimizer files and raw
GPU traces are excluded.

The [RTX deployment test](docs/rtx-validation.md) records a successful fresh Docker build, 59 ms median public-sample serial latency and 34.7K input tokens/s in warmed concurrent bursts. Cross-GPU and cross-batch probability differences remain disclosed; this is not private leaderboard acceptance.

## Decision Index submission

The model, results and code are public. [Submission PR #227](https://github.com/apolinario/decision-index/pull/227) is open for review; leaderboard inclusion and private quality/latency checks remain pending.
