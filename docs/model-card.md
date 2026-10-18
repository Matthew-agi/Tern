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

Use the included [runtime](../README.md). A stock Transformers causal-language-model loader is not sufficient for the trained head. The runtime exposes `/v1/systemone`, including every supplied question and option. Capacity is 32,768 encoded tokens for the complete state and question set; oversized inputs are rejected, never truncated.

## Public evaluation

The complete Decision Index 0.3 public evaluation scores **59.10** (raw index **69.04**): **140,176 successful requests and two unsupported BRIGHT requests**, out of 140,178. All 37 index benchmarks and six auxiliary benchmarks are included. The official scorer reports `complete: true`; an independent local rescore reproduced every score field. [Results repository](https://huggingface.co/datasets/matthewagi/Tern-35B-A3B-Decision-Index). This is not an official Full score or accepted leaderboard placement: maintainers evaluate the private portions and single-request latency themselves.

| Auxiliary benchmark | Requests | Metric | Result |
|---|---:|---|---:|
| MMLU | 14,033 | Accuracy | 85.71% |
| ARC-Easy | 2,376 | Accuracy | 98.99% |
| ARC-Challenge | 1,172 | Accuracy | 96.84% |
| RouterBench | 10,000 | Selected quality | 0.7954 |
| SGD/SGD-X | 2,500 | Macro-F1 | 0.1071 |
| SimpleBench | 10 | Accuracy | 30.00% |

Auxiliary metrics are excluded from the headline index. The evaluation combines the original 127,668-request run with the remaining 12,510 auxiliary requests on the same frozen weights and packaged runtime; no original predictions were replaced. All request IDs and recomputed payload hashes match the pinned complete public suite. Scorer revision and reconstruction details accompany the results.

The public score has development-exposure and training-overlap limitations. See [training disclosure](training-disclosure.md). Do not describe it as an uncontaminated held-out result.

## Reproducibility and limitations

Tested on one NVIDIA A100-SXM4-80GB. The isolated package installation exactly reproduced the previous environment's serial probabilities on 65 requests / 586 fields; repeated serial execution was identical. Changing serial execution to concurrent batches changed 2/586 decisions, with maximum probability difference 0.118828 and mean difference 0.003615. This fails the original cross-batch probability-drift screen. No relaxed tolerance or cross-batch numerical-parity claim is made. The original native-training versus serving differences are separately documented in the result artifacts.

Fresh vLLM/Triton/Inductor/extension caches compiled successfully and reproduced the serial reference probabilities exactly. A separate, explicitly warmed sample of 128 natural-length public requests on A100 measured median **62.65 ms**, mean **87.65 ms**, p80 **118.67 ms** and p95 **176.03 ms**, including HTTP, encoding and inference, excluding loading and warmup. All responses were identical between warmup and measurement.

Concurrent public evaluation uses 128 clients. Its timings are not the board's single-request measurement. RTX PRO 6000 compatibility and the board's private latency gate have not been verified on that hardware.

The model is an experimental decision scorer. Confidence values are learned probabilities, not guarantees of correctness. It is not a generative assistant or multimodal release.

## Provenance and attribution

Base revision: `Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0`. The owner has selected Apache-2.0 for Tern. The base model's Apache-2.0 license and attribution are preserved, as is the CLEF-derived runtime's original notice. Training-source evidence remains in the disclosure; this license choice does not rewrite source dataset terms. This GitHub package does not itself change the privately staged Hub repository or submit a benchmark PR.

The downloaded `export.json` records the checkpoint and training/validation fingerprints. The packaged [release manifest](../src/tern/configs/release.json) records all inference-file hashes and bytes. Optimizer state, training data, benchmark inputs and private traces are not included. GPU results above come from the original A100 package; this new launcher and the RTX/H100 final-checkpoint deployments still require hardware validation.
