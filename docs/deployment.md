# Hardware profiles and vLLM conversion

By default, every profile consumes the same immutable BF16 model snapshot and trained head:
`matthewagi/Tern-35B-A3B@05db86f1d322fd1eec865f769a04e6ddeabb8187`.
An explicit `--adapter top4-50m` selects a separately verified CPU merge. See
[execution policies](execution-policies.md) for pruning, exit and trained-model commands.

| Setting | RTX PRO 6000 | A100 | H100 fastest | H100 BF16 control |
|---|---|---|---|---|
| Quantization | none | none | `fp8_per_channel` | none |
| Recurrent prefill | Triton | Triton | FlashInfer | Triton |
| Head batch records / token budget | 16 / 16,384 | 16 / 16,384 | 32 / 32,768 | 16 / 16,384 |
| Packed head projections | off | off | on | off |
| Fused expert activation/quantization | off | off | on | off |
| MoE tile file | backend default | A100 SXM4 80GB | H100 PCIe FP8 | backend default |
| Batch token budget | 32,768 | 32,768 | 65,536 | 32,768 |
| Compiled/vectorized head and fused conv/norm/shared quant | off | off | on | off |
| Suggested client concurrency | 128 to test | 128 | 256 | 128 to test |

Shared settings are BF16 model dtype, FlashAttention, Triton MoE, 32K context, 256 sequences, 16K piecewise graph limit, no runtime LoRA, stateless
complete prefill and single-request submission. Frontend connection capacity is
128, including on H100: this preserves the recorded run. The H100 client had 256
outstanding requests; those are different limits. These are throughput settings;
serial latency is measured separately.

## What “conversion” means

For RTX/A100, vLLM loads the merged BF16 tensors directly. The pooling plugin,
custom head, metadata transport and source-checked stateless hooks are all
included in this package. A stock `vllm serve` command without those components
is not a complete Tern server.

For H100, `--quantization fp8_per_channel` asks the pinned vLLM backend to convert
eligible BF16 backbone linear/expert weights into its FP8 runtime representation
while loading. The opt-in activation-fusion patch produces the representation
expected by those kernels. The HF checkpoint on disk stays unchanged; startup
conversion repeats when starting a new process. This is the measured H100 path,
not an INT4 export or Kestrel model conversion. The custom head retains its
original precision policy; packed projections change execution grouping.

The updated H100 profile integrates the final step-1674/continuation campaign.
[Historical results and numerical boundaries](execution-policies.md) are retained;
the packaged branch still needs GPU acceptance testing. No profile is silently
substituted when a kernel or memory check fails.

## Exact commands and caches

`tern plan --profile PROFILE` prints the exact vLLM arguments. The launch log also
records the selected GPU UUID, profile, model revision and command. Keep that log
with benchmark measurements. Model downloads share one HF cache; compiler caches
are separate by profile, artifact revisions and execution policy under `TERN_CACHE_DIR`. The configured tile filename must
match the GPU name; a different A100/H100 form factor uses backend default tiles
and requires its own retest. No A100/H100 tile override is forced onto Blackwell.

The launcher clears inherited `CLEF_*`, `VLLM_*`, `KESTREL_*` and
`TORCH_COMPILE_*` and `TERN_H100_*` experiment flags before setting the selected
configuration. Hardware settings remain in profiles; `--experts`,
`--normalization-reference` and `--exit-layer` select execution independently.
The default is eight active experts and full depth. The launch log and `/ready`
identify the effective policy and artifact.

The package pins vLLM 0.30.0 (CUDA 12.9), Torch 2.13.0+cu129, Triton 3.7.1,
Transformers 5.19.0 and FlashInfer 0.6.18.post1. Direct numerical dependencies
match the frozen release. Transitive dependencies were newly resolved into
`requirements.lock`; that resolver success is not a GPU integration test.

The model file manifest pins all 33 inference files, totaling 69,778,082,886
bytes, including config/tokenizer/head. An existing local snapshot is accepted
only if it matches. HF README/license metadata are not part of the inference
allowlist, so a model-card edit cannot silently change the served tensors.

The RTX profile pins `CUBLAS_WORKSPACE_CONFIG=:4096:8` before worker startup. This stabilized serial outputs across fresh processes in the [reproducibility investigation](numerical-reproducibility.md). It does not guarantee cross-batch or cross-GPU identity; A100 defaults remain unchanged; H100 now pins the same workspace setting used in its final campaign.
