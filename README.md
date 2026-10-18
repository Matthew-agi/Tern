# Tern

Typed decisions with probabilities, served by the trained joint head on a
Qwen3.6-35B-A3B backbone. One request supplies a state and named questions with
their complete answer options. Tern returns every decision in one forward pass;
it does not generate text.

This repo packages the final **step-1674, 40-layer, top-8** model's frozen runtime.
The numerical Python files are byte-preserved and checked in CI. Weights and the
trained head live together on [Hugging Face](https://huggingface.co/matthewagi/Tern-35B-A3B),
pinned by revision and SHA-256. Code is Apache-2.0; the owner has selected
Apache-2.0 for Tern. Original dependency notices and [training disclosures](docs/training-disclosure.md)
are retained. This repo does not change Hub visibility or submit benchmark results.

**Prepared for RTX PRO 6000 Blackwell 96GB; GPU validation of this new package is
pending.** A100/H100 profiles reproduce recorded launch settings, with the
evidence limits below. This package has not yet been built with Docker or tested
on the target GPU. A working local CPU test suite is not a GPU compatibility claim.

## Start on RTX PRO 6000

Use a Linux x86_64 host with the 96GB Blackwell card, a CUDA-12.9-compatible NVIDIA
driver, Docker Compose v2 and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Plan for at least **120GB free disk** for the 69.8GB model, image and compilation
caches; **128GB host RAM** is a provisioning recommendation, not a measured minimum.
The 48GB RTX 6000 Ada is a different card and is not supported by this BF16 profile.

```bash
git clone https://github.com/Matthew-agi/Tern.git
cd Tern
# While the model is private, provide a read token with access to its HF repo.
read -r -s -p "Hugging Face read token: " HF_TOKEN; export HF_TOKEN; echo
docker compose up --build
```

Git authentication is also needed while this GitHub repo is private. On first
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

| Profile | Execution | Historical input throughput | Evidence boundary |
|---|---|---:|---|
| `rtx-pro-6000` (default) | BF16, Triton recurrent kernels, original head batching | Unmeasured | Blackwell 96GB validation pending |
| `a100` | BF16, A100 tuned MoE tiles | 24,514 tok/s matched panel | A100 SXM4 80GB, final step 1674; measured before this repackaging |
| `h100` | Load-time FP8, FlashInfer recurrent kernels, fused activation quantization, packed head, H100 tuned tiles | 45,010 tok/s matched panel; 47,109 sustained | H100 PCIe 80GB, **step 624**; not final-checkpoint throughput or parity |
| `h100-bf16` | BF16 reference with original head batching | Unmeasured | Control for checking the H100 FP8 profile on final weights |

All use 32,768 batch/context tokens, 256 sequences, 16,384-token piecewise CUDA
graphs and stateless complete-prefill execution. The known winning client loads
are 128 concurrent requests on A100 and 256 on H100. Client concurrency is not a
server flag. See [exact settings and conversions](docs/deployment.md) and
[recorded evidence](docs/profile-evidence.json).

The H100 profile keeps the architecture, head, depth and expert count, but **FP8
conversion and packed-head arithmetic can change probabilities**. It is an
explicit personal performance-test profile. Default RTX/A100 execution retains
the frozen BF16 path. No pruning, early exit, Kestrel replacement, result cache
or unqualified kernel optimization is enabled.

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

The [model card](docs/model-card.md) records the completed public Decision Index
evaluation: **59.10 public / 69.04 raw**, 140,176 successful of 140,178 requests.
This is not an accepted leaderboard placement or the private Full score.
Cross-batch numerical variation and training/development overlap are disclosed;
we do not promise identical probabilities across batches or GPU types.

Directory layout:

```text
src/tern/          CLI, lifecycle, profiles, artifact verification, benchmark client
src/kev_vllm/      Frozen model, joint head, encoding, kernels and vLLM plugin
src/serve_clef_api.py  Frozen SystemOne API; Tern wrapper changes only service identity
examples/          Synthetic request (no benchmark inputs)
tests/             CPU contracts and packaging checks
docs/              Architecture, GPU testing, provenance and disclosures
```

See [architecture](docs/architecture.md) for how the backbone, trained head and
API connect. Large weights remain on HF; code, launch profiles and compact
provenance belong here. Training data, benchmark inputs, optimizer files and raw
GPU traces are excluded.
