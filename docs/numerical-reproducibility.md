# Numerical reproducibility investigation — 2026-10-18

## Scope and change

The RTX profile now sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the worker environment, before starting vLLM and the API worker. This overrides an inherited workspace setting for this profile. A100/H100 profiles are unchanged. No tensor, tokenizer, trained head source, routing policy, precision policy, layer count, or numerical acceptance threshold changed.

The original deployment used code `2e95f9cf155f7fb1f393e48fe43f2281d61b5b85` and the same immutable step-1674 snapshot. All tests used the same 128 public requests, 344 decisions and exact payload hashes. Diagnostic traces were taken separately from production timing. The GPU is RTX PRO 6000 Blackwell 96GB, driver 580.126.09; the pinned CUDA 12.9 dependencies remain unchanged.

## Findings

- The unpinned path reproduced differing probabilities across fresh processes, despite exact repeated serial predictions within each process. Two instrumented fresh processes differed by up to **0.057559**, changing one selected answer.
- All **123 head parameters/buffers** matched across those processes and matched the exported checkpoint at their intended serving dtypes. No head-loading or dropout defect was found.
- Recorded FLA/Triton autotuner selections matched. Backbone hidden states already differed before entering the head.
- With `:4096:8`, **three fresh diagnostic processes produced exactly identical probabilities**, also matching the original RTX deployment's first run. Two of these captured all 30 recurrent blocks, all 10 full-attention blocks and the final hidden states for a fixed 131-token input: all 41 captures matched exactly.
- An unpinned rollback happened to match the fixed run. The issue is intermittent; not every unpinned restart drifts. These tests establish a reproducible mitigation on this panel, not a universal determinism guarantee.
- As a controlled test, setting the workspace to `:16:8` changed three selected answers, with maximum probability difference **0.090506** versus `:4096:8`. In its trace, differences already appeared in the first layer's projection outputs, before the recurrent calculation. This directly demonstrates that workspace-dependent GEMM execution can change BF16 rounding and propagate through the backbone.

NVIDIA documents workspace configuration as a remedy for cuBLAS reproducibility problems involving concurrent streams/internal workspace selection. Its reproducibility guarantee is restricted to a matching architecture/SM count and toolkit; this is not a promise of bitwise identity between A100 and Blackwell. See [cuBLAS 12.9 results reproducibility](https://docs.nvidia.com/cuda/archive/12.9.1/cublas/index.html#results-reproducibility). Our experiments isolate a workspace-sensitive numerical path, but do not establish which internal cuBLAS algorithm every unpinned restart selected.

## Cross-GPU comparison

The fixed RTX result still differs from the saved A100 serial reference: **4/344 selected answers**, maximum probability difference **0.275133**, mean absolute difference **0.004851**. Repeating the RTX run with the A100 MoE tile configuration (confirmed loaded in the worker log) produced exactly the same RTX probabilities. The tile override was not adopted.

Weights and input token counts match. The remaining hardware-specific arithmetic cannot be localized to the first differing A100 operation because an equivalent A100 layer trace is unavailable. It is not justified to claim that the workspace change solves cross-GPU parity. The diagnostic harness retains matching layer-capture support for a future A100 comparison; no new GPU was rented for this investigation.

## Labeled smoke evaluation

The 128-request panel contains 112 requests / 314 ordinary hard-label choice or noul decisions after excluding routing-outcome, chess-reference, group-ranking and unscored fields. This is a small public smoke subset, **not a Decision Index score** or a general equivalence test.

| Variant | Correct | Accuracy | Brier |
| --- | ---: | ---: | ---: |
| Saved A100 serial | 273/314 | 86.94% | 0.090463 |
| Original RTX / fixed-workspace RTX | 272/314 | 86.62% | 0.091464 |

Brier here is one-half of the summed multiclass squared error, equal to ordinary binary Brier for two options. The paired request-bootstrap 95% interval for the accuracy difference is approximately **−1.33 to 0 percentage points**; for the Brier increase of **0.001001**, approximately **−0.000897 to +0.004315**. The small sample supports similar observed quality, not proof of equivalent quality across the full suite.

## Verification status

The launcher change passes all **29 CPU tests**, including inherited-workspace override coverage. The rebuilt production image, with all diagnostic hooks removed, matched the original RTX probabilities exactly and remained exact after concurrent load. This is a fourth fresh process on the fixed path. Private requests, response records and tensor traces remain outside this repository.


The three warmed matched 257-request throughput runs measured **34,889.5 / 34,788.3 / 34,683.6 input tokens/s**, versus the original mean of 34,666.5. This shows no observed performance penalty; the small difference is within ordinary run variation and is not claimed as a speedup. Startup, compilation and idle time are excluded. Serial versus concurrent output variation remains: **3/344 selected answers**, maximum probability difference **0.086051**. Serial predictions before and after that load remained exactly identical.

The fixed container is healthy and retained for follow-up. No diagnostic mounts, monkeypatches or HF token remain in the production container. The original failure evidence, controls, smoke scoring code and private trace captures are retained in the investigation directory outside GitHub.

The subsequent [complete RTX public evaluation](decision-index-rtx.md) extends this smoke check to the full suite and compares it with the original A100 results.
