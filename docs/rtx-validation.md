# RTX PRO 6000 deployment validation — 2026-10-18

**Historical initial deployment report.** The later [full RTX evaluation](decision-index-rtx.md) and [workspace investigation](numerical-reproducibility.md) supersede the pending full-evaluation and restart status below. Original measurements are retained for provenance.

> Follow-up: the RTX launcher now pins cuBLAS workspace selection. See the [restart investigation and smoke quality check](numerical-reproducibility.md). The initial observations below are retained as historical evidence.

Tested code: `2e95f9cf155f7fb1f393e48fe43f2281d61b5b85`, downloaded as a fresh GitHub archive.
Model snapshot: `05db86f1d322fd1eec865f769a04e6ddeabb8187`; all 33 inference files verified against the release SHA-256 manifest. Full 40 layers, eight active experts, merged BF16 weights and the unchanged trained head.

Hardware: one RTX PRO 6000 Blackwell, 97,887 MiB, driver 580.126.09, 16 vCPU; Ubuntu 22.04 with Docker and NVIDIA Container Toolkit. Rental rate: $1.35/hour. Image: `sha256:c1a68a2ca83c18a47fe653e9f3a759756e3b454b98dabde1243b68ac2a1833a2`.

## Installation and interface

The documented Docker image built on the first attempt and passed the installed vLLM source contracts. Twelve applicable GPU kernel tests passed without tolerance changes; two H100-only FlashInfer tests were skipped. Model loading and graph capture succeeded without eager mode or runtime source changes. Some Triton autotuning candidates exceeded Blackwell shared memory; autotuning selected supported candidates and startup completed.

The official Decision Index HTTP engine and validator at `apolinario/decision-index@f8552a13b643d39f61185f5951bb5fed16bedf51` accepted choice and noul responses, multiple decisions and complete option distributions. Explicit HTTP batches and the additional score API passed. Overlength input was rejected as official `Unsupported`, without truncation; an empty schema returned HTTP 422. Readiness followed successful inference.

A clean shutdown released all GPU memory. The container was recreated with the HF token removed from both its environment and the host `.env`; the verified cached model loaded and the service became healthy. The rental is retained for follow-up testing.

## Warmed measurements

The 128-request latency panel is the same deterministic natural-length public sample used in the final A100 latency check. Each whole panel was warmed before measurement. These are HTTP wall times, including encoding and inference.

| Measurement | Result |
| --- | ---: |
| Serial requests completed | 128/128 in both measured repeats |
| Median | 59.35 / 59.27 ms |
| Mean | 73.57 / 73.56 ms |
| p80 | 88.93 / 88.74 ms |
| p95 | 133.43 / 133.53 ms |
| Maximum | 348.45 / 349.70 ms |
| Same-panel concurrency 128 | 33,819 input tokens/s; mean request latency 2,038 ms |
| Explicit batches of 16, concurrency 4 | 33,502 input tokens/s; mean batch completion 1,335 ms |

The separate throughput panel has 257 requests and 580,694 encoded input tokens, with lengths 139–32,677. After one complete warmup, three concurrency-128 runs measured **34,821.94 / 34,483.44 / 34,694.16 tokens/s**, averaging **34,666.52 tokens/s**. GPU-only cost is **$0.01082 per million encoded input tokens**, excluding build, download, compilation, warmup and idle time. These short repeated bursts are not a sustained-load result. Observed sampled peak GPU allocation was 78,241 MiB.

A separate 14-request natural-length diagnostic included inputs near the 32K limit. All succeeded; maximum serial latency was 1,096 ms. This deliberately length-stratified panel is not the random latency distribution.

## Numerical boundaries and submission status

Repeated RTX serial runs were exactly identical: 128 requests, 344 decisions, zero probability differences or selected-answer changes.

Across the clean cached restart, all 344 selected answers stayed the same, but probabilities shifted by up to 0.030510 (mean absolute difference 0.001676). Two repeated serial passes within the restarted process were again exactly identical. Cross-process probability reproducibility therefore also remains unresolved. No source or parameter change was made during restart.

Cross-GPU equivalence **did not pass**: against the saved A100 serial outputs for the same payload hashes and token counts, 4/344 decisions changed; maximum probability difference was 0.275133 and mean absolute probability difference was 0.004851. This is not an accuracy measurement. The runtime sources and weights were preserved. GPU arithmetic and the RTX default versus A100 tuned MoE tiles are possible explanations; this test does not isolate their contributions.

RTX serial versus concurrency-128 changed 2/344 decisions (maximum probability difference 0.081891, mean 0.003882). Explicit batches also changed 2/344 decisions (maximum 0.083312, mean 0.003674). These exceed the historical numerical screen; no tolerance was relaxed and no claim of batch invariance is made.

The public latency sample is comfortably below the Index's median/mean/p80 1,000 ms limits. It does **not** certify acceptance on the maintainers' private sample. No full RTX quality evaluation was run, and the prior A100 public score is not an RTX score. Maintainers must have access to runnable artifacts and independently check private quality and latency. See [official submission requirements](https://github.com/apolinario/decision-index#submitting-a-model-to-the-leaderboard).

Private inputs and full response records are retained outside this code repository. No benchmark payloads, model weights, credentials or server access details are published here.
