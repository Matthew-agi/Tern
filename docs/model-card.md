---
license: apache-2.0
base_model: Qwen/Qwen3.6-35B-A3B
tags:
- decision-index
- mixture-of-experts
- vllm
- calibrated-decisions
language:
- en
---

# Tern-35B-A3B

Tern is a typed decision model: supply a state and named questions with their complete answer options, and receive a probability distribution for every question. It uses a Qwen3.6-35B-A3B backbone with a jointly trained decision head. It does not generate reasoning or text tokens.

This release is optimizer checkpoint 1674, trained on 499,999,909 tokens. It contains merged training-rounded BF16 backbone/adapter weights and a trained head stored in FP32. The frozen serving path precasts selected head parameters to BF16 and uses BF16 autocast. The default and frozen public evaluation use all 40 layers and eight routed experts, with no early exit, expert pruning, external tools, retrieval or test-time augmentation. Optional serving policies can reduce the routed expert count or exit unconditionally at an earlier layer; these deliberately change predictions and are not the configuration behind the full public scores below.

Use the tested [Tern Docker serving repository](https://github.com/Matthew-agi/Tern) for RTX PRO 6000, A100 and H100 profiles. The default RTX BF16 profile includes the cuBLAS workspace reproducibility setting automatically; no manual patch is required. The model, results and serving code are public; no access token is required. The original [frozen A100 runtime](https://huggingface.co/matthewagi/Tern-35B-A3B/blob/main/runtime/README.md) is retained for provenance; the dedicated repository is the RTX submission entrypoint. The optimized H100 FP8 profile and optional policies are available on `main` and have been tested through a fresh Docker deployment; they remain separate from the frozen RTX public evaluation. A stock Transformers causal-language-model loader does not implement the trained decision head. Capacity is 32,768 encoded tokens for the entire state and all questions/options; oversized requests are rejected without truncation.

## Download and serve

On a Linux GPU host with Docker Compose v2, a compatible NVIDIA driver and the configured [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):

```bash
git clone https://github.com/Matthew-agi/Tern.git
cd Tern
# Default: RTX PRO 6000 Blackwell 96GB, original 8/8, all 40 layers.
docker compose up --build
```

Use `TERN_PROFILE=h100` for H100 or `TERN_PROFILE=a100` for A100. Plan for at least 120GB free disk for the base deployment; additional adapter merge caches require more space. The supported RTX profile targets the 96GB Blackwell card, not the 48GB Ada card. Wait for `Tern READY` or a healthy container. Model files are downloaded and hash-verified automatically; a stock Transformers causal-LM loader does not implement this decision head.

```bash
curl -fsS http://localhost:8000/ready
curl -fsS http://localhost:8000/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @examples/request.json
```

Stop the current service before switching. The following use the public original weights:

```bash
TERN_PROFILE=h100 docker compose up --build
TERN_PROFILE=h100 TERN_EXPERTS=4 docker compose up --build
TERN_PROFILE=h100 TERN_EXPERTS=4 TERN_EXIT_LAYER=30 docker compose up --build
```

`TERN_EXPERTS` selects the number of dynamically routed experts per token, from 1 to 8; it does not select fixed expert IDs. Normalization reference stays 8 by default. `TERN_EXIT_LAYER=30` uses the first 30 layers and the original final norm and head, unconditionally. These controls are also exposed by `tern serve --profile h100 --experts 4 --exit-layer 30`. A100/RTX support the controls, but new hardware/policy combinations need their own GPU validation.

The separate trained 50M-token 4/8 continuation is **private**. Authorized users can supply an HF read token in an untracked, restricted-permission `.env` file and use `TERN_ADAPTER=top4-50m`; the launcher automatically downloads and merges its pinned LoRA and trained head into a separate cache. Public pruning alone needs no token and does not load that continuation. See [execution policies](https://github.com/Matthew-agi/Tern/blob/main/docs/execution-policies.md).

## Packaged H100 throughput

Fresh H100 **SXM 80GB** deployment, serving commit `ec475b598306ebe29ea019533e6c6f78821d6674` (now on `main`):

| Configuration | Input tok/s | GPU $/M at $2.35/hour |
|---|---:|---:|
| Original 8/8, 40 layers | 80,173 | $0.00814 |
| Original pruned 4/8, 40 layers | 91,875 | $0.00711 |
| Original pruned 4/8, exit 30 | 121,267 | $0.00538 |
| Trained 4/8, 40 layers | 95,244 | $0.00685 |
| Trained 4/8, exit 30 | 115,498 | $0.00565 |

Means of all three warmed repetitions of 257 requests / 580,694 encoded input tokens, concurrency 256. These short bursts include HTTP, encoding and inference, but exclude loading, merging, compilation warmup and idle time. $2.35/hour is a comparison assumption, not a rental quote. Effective costs increase at lower productive utilization. Historical H100 PCIe measurements are separate and should not be treated as matched hardware comparisons.

No new quality evaluation was run in this deployment test. Serial repeats matched exactly, but concurrent probabilities varied, with a maximum absolute difference of 0.213 across the five configurations. The automatic merge's 122 head tensors were verified exactly against the continuation checkpoint. See [full deployment scope, host prerequisites and limitations](https://github.com/Matthew-agi/Tern/blob/main/docs/h100-validation.md).

## Public evaluation

| Complete public 0.3 run | Public index | Raw index | Requests |
| --- | ---: | ---: | ---: |
| RTX PRO 6000, fixed-workspace Docker | 59.01 | 68.93 | 140,178 |
| Original A100 | 59.10 | 69.04 | 140,178 |

The fresh RTX run includes all 37 index and six auxiliary benchmarks: 140,176 successful requests, 2 unsupported BRIGHT requests, and zero errors. The official scorer reports `complete: true`, and an independent local rescore reproduces every score field. All request IDs and payload hashes match the frozen suite. Complete predictions, exact settings, per-benchmark metrics and comparisons are in the [results repository](https://huggingface.co/datasets/matthewagi/Tern-35B-A3B-Decision-Index), under `runs/step1674-rtx-top8/`. The older A100 run remains unchanged under `runs/step1674-top8/`.

These are historical public-run scores, distinct from separately published Full scores or leaderboard evaluations. See [TRAINING_DISCLOSURE.md](https://huggingface.co/matthewagi/Tern-35B-A3B/blob/main/TRAINING_DISCLOSURE.md) for public development exposure and training-overlap candidates; this is not an uncontaminated held-out claim.

## Reproducibility and limitations

The RTX launcher pins `CUBLAS_WORKSPACE_CONFIG=:4096:8`. The fixed serial panel matched across four fresh processes; head tensors and numerical source files were unchanged. This mitigates observed restart variation on that panel, not a universal determinism guarantee. Cross-GPU and cross-batch probability equivalence remain unproven. In the matched full public runs, RTX versus A100 changed 4,384/312,256 selected decisions (1.404%). Both hardware and batch scheduling differ; these effects are not separately isolated. No numerical tolerance was widened.

A separately warmed 128-request RTX serial panel after the full evaluation measured median 59.48 ms, mean 74.09 ms, p80 89.38 ms and p95 134.59 ms. These include HTTP, encoding and inference, and exclude model loading and warmup. The concurrent full-evaluation timings include queueing and are not single-request latency. These measurements do not establish private benchmark quality or latency results.

The model is an experimental decision scorer. Its probabilities are not guarantees of correctness. It is not a generative assistant or multimodal release.

## Provenance and attribution

Base revision: `Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0`. Tern model weights, the trained head and bundled Tern code are licensed under [Apache-2.0](https://huggingface.co/matthewagi/Tern-35B-A3B/blob/main/LICENSE). The base model's original license is preserved in [BASE_MODEL_LICENSE](https://huggingface.co/matthewagi/Tern-35B-A3B/blob/main/BASE_MODEL_LICENSE), and the CLEF-derived runtime retains its [original notice](https://huggingface.co/matthewagi/Tern-35B-A3B/blob/main/runtime/kev_vllm/CLEF_LICENSE). See [NOTICE](https://huggingface.co/matthewagi/Tern-35B-A3B/blob/main/NOTICE) and [TRAINING_DISCLOSURE.md](https://huggingface.co/matthewagi/Tern-35B-A3B/blob/main/TRAINING_DISCLOSURE.md) for attribution and training-source evidence. The license choice does not rewrite source dataset terms.

`export.json` records the checkpoint and training/validation fingerprints. `model-file-sha256.json` records the inference-file bytes. Optimizer state, training data, benchmark inputs and private traces are not included.

## Upstream SGD auxiliary-input correction

A final upstream check found [commit `2d7791c`](https://github.com/apolinario/decision-index/commit/2d7791c56b3eb98e4940efc056fba28032cad54e), which fixes an SGD builder bug: rows previously retained a reference to the growing dialogue history and therefore included future turns. All 2,500 SGD requests in both frozen full runs are affected. Their original SGD scores should not be interpreted as valid current-turn intent classification results.

A separate supplemental check keeps the same selected request IDs, schemas, labels, model and head, but stops each history at its recorded current user turn, matching the corrected builder's snapshot semantics. All 2,500 requests succeeded. **Corrected SGD macro-F1 is 0.7033**, versus **0.1052** in the frozen RTX run (original A100: 0.1071). Independent local rescoring reproduced the corrected benchmark summary exactly. [Supplemental results](https://huggingface.co/datasets/matthewagi/Tern-35B-A3B-Decision-Index/blob/main/diagnostics/sgd-history-fix/summary.json) include provenance and separate predictions.

SGD is excluded from the headline index: **59.01 public / 68.93 raw remains unchanged**. The complete canonical frozen runs and their payload hashes are preserved untouched. The supplemental corrected requests have different payload hashes and are not silently substituted into those runs. Upstream HEAD checked was `f5a6e4fd85fed05e77878dfc4966d0568bfb45ec`; its scoring/edition files and published suite hash declarations are unchanged from the frozen scorer pin. This discrepancy is disclosed for maintainer review. No benchmark input text is redistributed.

## Submission status

The model, results and serving code are public. [Decision Index submission PR #227](https://github.com/apolinario/decision-index/pull/227) is closed without merging (checked October 22, 2026). This PR status alone does not establish leaderboard inclusion or private evaluation results; consult the current Decision Index separately. The evaluated inference files and complete predictions are unchanged.
