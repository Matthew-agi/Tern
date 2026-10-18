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

This release is optimizer checkpoint 1674, trained on 499,999,909 tokens. It contains merged training-rounded BF16 backbone/adapter weights and a trained head stored in FP32. The frozen serving path precasts selected head parameters to BF16 and uses BF16 autocast. All 40 layers and eight routed experts are active. No early exit, expert pruning, external tools, retrieval or test-time augmentation is used.

Use the tested [Tern Docker serving repository](https://github.com/Matthew-agi/Tern/tree/9723b745367735473f03c93a6f3639c4fb2c8c3f) for RTX PRO 6000, A100 and H100 profiles. The default RTX BF16 profile includes the cuBLAS workspace reproducibility setting automatically; no manual patch is required. GitHub and HF access are required while these repositories are private. The original [frozen A100 runtime](https://huggingface.co/matthewagi/Tern-35B-A3B/blob/main/runtime/README.md) is retained for provenance; the dedicated repository is the RTX submission entrypoint. The optional H100 FP8 profile is a separate experiment. A stock Transformers causal-language-model loader does not implement the trained decision head. Capacity is 32,768 encoded tokens for the entire state and all questions/options; oversized requests are rejected without truncation.

## Public evaluation

| Complete public 0.3 run | Public index | Raw index | Requests |
| --- | ---: | ---: | ---: |
| RTX PRO 6000, fixed-workspace Docker | 59.01 | 68.93 | 140,178 |
| Original A100 | 59.10 | 69.04 | 140,178 |

The fresh RTX run includes all 37 index and six auxiliary benchmarks: 140,176 successful requests, 2 unsupported BRIGHT requests, and zero errors. The official scorer reports `complete: true`, and an independent local rescore reproduces every score field. All request IDs and payload hashes match the frozen suite. Complete predictions, exact settings, per-benchmark metrics and comparisons are in the [results repository](https://huggingface.co/datasets/matthewagi/Tern-35B-A3B-Decision-Index), under `runs/step1674-rtx-top8/`. The older A100 run remains unchanged under `runs/step1674-top8/`.

These are public scores, not the private Full score or an accepted leaderboard placement. See [TRAINING_DISCLOSURE.md](training-disclosure.md) for public development exposure and training-overlap candidates; this is not an uncontaminated held-out claim.

## Reproducibility and limitations

The RTX launcher pins `CUBLAS_WORKSPACE_CONFIG=:4096:8`. The fixed serial panel matched across four fresh processes; head tensors and numerical source files were unchanged. This mitigates observed restart variation on that panel, not a universal determinism guarantee. Cross-GPU and cross-batch probability equivalence remain unproven. In the matched full public runs, RTX versus A100 changed 4,384/312,256 selected decisions (1.404%). Both hardware and batch scheduling differ; these effects are not separately isolated. No numerical tolerance was widened.

A separately warmed 128-request RTX serial panel after the full evaluation measured median 59.48 ms, mean 74.09 ms, p80 89.38 ms and p95 134.59 ms. These include HTTP, encoding and inference, and exclude model loading and warmup. The concurrent full-evaluation timings include queueing and are not single-request latency. Maintainers must still run their private quality and latency checks.

The model is an experimental decision scorer. Its probabilities are not guarantees of correctness. It is not a generative assistant or multimodal release.

## Provenance and attribution

Base revision: `Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0`. Tern model weights, the trained head and bundled Tern code are licensed under [Apache-2.0](../LICENSE). The base model's original license is preserved in [BASE_MODEL_LICENSE](BASE_MODEL_LICENSE), and the CLEF-derived runtime retains its [original notice](../src/kev_vllm/CLEF_LICENSE). See [NOTICE](../NOTICE) and [TRAINING_DISCLOSURE.md](training-disclosure.md) for attribution and training-source evidence. The license choice does not rewrite source dataset terms.

`export.json` records the checkpoint and training/validation fingerprints. `model-file-sha256.json` records the inference-file bytes. Optimizer state, training data, benchmark inputs and private traces are not included.
