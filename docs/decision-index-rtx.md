# RTX PRO 6000 full public evaluation

The unchanged step-1674 model completed all **140,178 public Decision Index 0.3 requests** on one RTX PRO 6000 Blackwell 96GB: **140,176 ok, 2 unsupported, zero errors**. All 37 index benchmarks and six auxiliary benchmarks are included. An independent local CPU rescore reproduced all score fields exactly except generation timestamps. Every request ID and payload hash matches the frozen complete public suite.

| Same checkpoint and trained head | Public index | Raw index |
| --- | ---: | ---: |
| A100, original full run | 59.10 | 69.04 |
| RTX, pinned workspace Docker | 59.01 | 68.93 |
| RTX minus A100 | -0.09 | -0.11 |

The A100 results are preserved unchanged in a separate run. These are public-panel scores, not private Full scores or an accepted leaderboard placement. Training/development exposure and overlap remain disclosed.

## Execution and performance

Tested code: [`9723b745367735473f03c93a6f3639c4fb2c8c3f`](https://github.com/Matthew-agi/Tern/tree/9723b745367735473f03c93a6f3639c4fb2c8c3f). Model inference snapshot: `05db86f1d322fd1eec865f769a04e6ddeabb8187`. All 40 layers, eight experts normalized to eight, merged BF16 and the original trained head. The default RTX profile sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` automatically before worker startup. No manual Docker patch or environment step is needed.

The complete evaluation consumed **147,595,988 encoded input tokens** in **70.1 minutes**: **35,075 tokens/s**, **$0.01069/M input tokens** at $1.35/hour. This is end-to-end concurrent evaluation at 128 clients, excluding model loading, initial compilation and idle time. It is a different workload from the matched 257-request warmed throughput bursts, which averaged 34,787 tokens/s.

After the evaluation, a separate warmed 128-request serial panel measured median **59.48 ms**, mean **74.09 ms**, p80 **89.38 ms**, and p95 **134.59 ms**. All 344 decisions reproduced the pre-evaluation fixed-workspace serial probabilities exactly. These public-sample timings do not certify the maintainers' private single-request latency gate.

## Numerical limits

The workspace-pinned path reproduced the same serial panel across four fresh processes in the preceding investigation. Full-suite RTX versus A100 changed **4,384/312,256 selected decisions (1.404%)**. Maximum probability difference was **0.631766**; mean absolute difference was **0.003120**. This comparison includes different hardware and concurrent batch scheduling; it does not isolate their individual effects. Every successful matched request used the same encoded token count. No tolerance was widened and no cross-GPU or cross-batch equivalence is claimed.

See [the restart investigation](numerical-reproducibility.md) for the workspace control, head hash checks and remaining batch sensitivity. The submission contains complete untouched predictions, unsupported responses, exact settings, scorer pin, source hashes and independent rescore evidence. The two over-capacity BRIGHT cases are rejected without truncation and remain in scoring.

## Reviewer startup

Use the pinned repository's normal `docker compose up --build` on a Linux x86_64 RTX PRO 6000 Blackwell 96GB host with a CUDA-12.9-compatible driver and NVIDIA Container Toolkit. The launcher downloads and SHA-verifies the 33 inference files, compiles, warms up and exposes `/v1/systemone`. The repositories are public; no access token is required. Use the BF16 RTX profile for this submission; H100 FP8 is a separate personal experiment.

No benchmark inputs, labels, model weights, credentials or raw GPU traces are included in this code repository. The public suite and private test requirements are described by the [official submission rules](https://github.com/apolinario/decision-index#submitting-a-model-to-the-leaderboard).

## Upstream SGD auxiliary-input correction

A final upstream check found [commit `2d7791c`](https://github.com/apolinario/decision-index/commit/2d7791c56b3eb98e4940efc056fba28032cad54e), which fixes an SGD builder bug: rows previously retained a reference to the growing dialogue history and therefore included future turns. All 2,500 SGD requests in both frozen full runs are affected. Their original SGD scores should not be interpreted as valid current-turn intent classification results.

A separate supplemental check keeps the same selected request IDs, schemas, labels, model and head, but stops each history at its recorded current user turn, matching the corrected builder's snapshot semantics. All 2,500 requests succeeded. **Corrected SGD macro-F1 is 0.7033**, versus **0.1052** in the frozen RTX run (original A100: 0.1071). Independent local rescoring reproduced the corrected benchmark summary exactly. [Supplemental results](https://huggingface.co/datasets/matthewagi/Tern-35B-A3B-Decision-Index/blob/main/diagnostics/sgd-history-fix/summary.json) include provenance and separate predictions.

SGD is excluded from the headline index: **59.01 public / 68.93 raw remains unchanged**. The complete canonical frozen runs and their payload hashes are preserved untouched. The supplemental corrected requests have different payload hashes and are not silently substituted into those runs. Upstream HEAD checked was `f5a6e4fd85fed05e77878dfc4966d0568bfb45ec`; its scoring/edition files and published suite hash declarations are unchanged from the frozen scorer pin. This discrepancy is disclosed for maintainer review. No benchmark input text is redistributed.

## Submission status

[Decision Index PR #227](https://github.com/apolinario/decision-index/pull/227) is open. Public release and submission are complete; maintainer review and private quality/latency checks remain pending.
