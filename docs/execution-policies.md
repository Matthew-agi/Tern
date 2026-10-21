# Expert count and early exit

Hardware, execution policy and weights are separate choices. The same CLI works
with `rtx-pro-6000`, `a100`, `h100` and `h100-bf16`; only `h100` enables the Hopper
FP8 kernel bundle. No flags means the original model, eight routed experts and
all 40 layers.

```bash
tern serve --profile h100 --experts 4
tern serve --profile h100 --experts 6
tern serve --profile h100 --exit-layer 30
tern serve --profile h100 --experts 4 --exit-layer 30
tern plan  --profile h100 --experts 4 --exit-layer 30
```

`--experts K` chooses **K dynamically routed experts per token in every layer**,
not particular expert IDs. All 256 expert weights remain stored, and the shared
expert remains active. Counts 1 through 8 are accepted. The default
`--normalization-reference 8` matches the pruning experiments: selected router
probabilities are divided by the sum of the top eight probabilities, rounded to
the gate dtype, and passed to the selected experts. A top-4 output therefore
retains its fraction of top-8 mass; it is not renormalized to sum to one.
The ordinary 8/8 policy keeps the stock vLLM router.

An explicit `--normalization-reference N` is also available, where K ≤ N ≤ 256.
Changing this changes the model computation. For H100 4/8 and 6/8, the measured
selection kernel preserves the pinned PyTorch selection/tie behavior. Other
combinations use native PyTorch selection. Counts 3, 5 and 7 and normalization
references other than 8 have no throughput/quality claims from this campaign.

`--exit-layer L` executes layers 1 through L inclusive, for every request. It
constructs only that prefix and skips the checkpoint's unused tail while loading.
It then applies the **original final norm and trained head** to the prefix output.
There is no confidence test or second head. The on-disk checkpoint remains full
depth. Values 1 through 40 are accepted; depth 30 is the measured reduced-depth
configuration. Other depths need their own quality and GPU validation. Pruning
and exit both intentionally change predictions; they are not numerical-parity
optimizations.

## The trained 4/8 continuation

`--experts 4` alone serves the original model with fewer active experts. To use
the 50M-token continuation and its trained head:

```bash
# HF_TOKEN must have read access to the private adapter repository.
tern download --adapter top4-50m
tern serve --profile h100 --adapter top4-50m
tern serve --profile h100 --adapter top4-50m --exit-layer 30
```

The adapter defaults to four active experts; an explicit `--experts` takes
precedence. It is pinned to
`matthewagi/Tern-35B-A3B-top4-ref8-50M-LoRA@7802373d15179be693b1156584d438666b591608`
and original Tern revision `05db86f1d322fd1eec865f769a04e6ddeabb8187`.
It is a custom trainable checkpoint, not a PEFT-format runtime adapter. The loader
downloads only the adapter tensors and cursor metadata, verifies SHA-256, merges
expert/shared-expert LoRA on CPU in FP32 and casts to the original BF16 weights.
All 522 trainable tensors, including the 122 head tensors, must be consumed.
The indexed head keeps its `pooler.head.*` keys. An incomplete head now prevents
serving, including the unprefixed-head export failure found during the campaign.

The merge is performed once into an atomic, separate cache, protected against
concurrent converters. It requires about **75GB additional free disk** beyond the
base model and image; budget roughly **200GB total**. Keep the recommended 128GB
host RAM. Subsequent starts verify the cached merge. CPU merging and load-time
FP8 conversion are excluded from the historical serving rates.

## Docker Compose

Set the same choices through environment variables; no image edits are needed.
Explicit native CLI flags take precedence over environment defaults.

```bash
TERN_PROFILE=h100 TERN_EXPERTS=4 TERN_EXIT_LAYER=30 docker compose up --build
# Trained continuation: provide HF_TOKEN in the shell or your untracked .env.
TERN_PROFILE=h100 TERN_ADAPTER=top4-50m TERN_EXIT_LAYER=30 docker compose up --build
# A100, original weights with six active experts and full depth:
TERN_PROFILE=a100 TERN_EXPERTS=6 docker compose up --build
```

`TERN_NORMALIZATION_REFERENCE` defaults to 8 and `TERN_EXIT_LAYER` to 40.
Unset `TERN_ADAPTER` and `TERN_EXPERTS` to restore the original eight-expert model.
Stop/recreate the service when changing choices. Compiler caches are isolated by
hardware profile, parent/adapter revisions and execution policy. `/ready` reports
the effective policy and artifact identity after the startup inference.

## H100 optimizations included

The `h100` profile now carries the final campaign's 65,536-token scheduling
budget, 16K piecewise capture limit, tuned Triton MoE tiles, stateless prefill,
FlashInfer recurrence, expert activation/FP8 fusion, fused convolution preparation,
GDN norm/quantization, shared-input quantization and selected rowwise FP8 dense
GEMMs. The head uses packed projections, compiled scoring/blocks, flattened KV
projection, vectorized option summaries and segmented span reductions. The
workspace setting is `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

The broad vLLM fusion switch that produced about 0.095 maximum probability drift
is excluded. Kestrel and unqualified CUTLASS expert replacements are excluded.
RTX/A100 default numerical paths and the dependency lock are retained. Patched
vLLM sources are SHA-checked; the optimized router also checks the exact PyTorch
build and SM90 hardware. Unsupported configurations fail instead of silently
selecting a different kernel stack.

| Historical H100 configuration | Public sample score | Input tok/s | GPU $/M at $2.35/h |
|---|---:|---:|---:|
| Original 8/8, depth 40 | 60.13 | 56,259 | 0.01160 |
| Original pruned 4/8, depth 40 | 56.29 | 65,891 | 0.00991 |
| Trained 4/8, depth 40 | 58.83 | 65,509 | 0.00996 |
| Trained 4/8, depth 30 | 55.41 | 84,475 | 0.00773 |

Evidence: [corrected depth-40 comparison](h100-trained4-depth40.json),
[depth-30 comparison](h100-trained4-exit30.json), [saved command](profile-evidence.json)
and [source provenance](h100-source-provenance.json). These are prior H100 PCIe
80GB measurements, **not a new benchmark of this packaged branch**. Quality uses
1,827 public development requests across 43 benchmarks, not the full/private
Index; continuation data permits replay. Throughput uses 257 requests / 580,694
encoded input tokens, concurrency 256 and three warmed repetitions, excluding
startup, compilation and idle time. No throughput transfer to RTX/A100 is claimed.

Local tests cover policy validation, exact prefix/tail handling, reference
routing, head equations, merge arithmetic, head-key preservation, immutable base
files, corruption rejection and interrupted-conversion cleanup. Before adopting
this branch, follow [GPU validation](testing.md) on the intended hardware with
the same model/policy and representative matched inputs. Saved baseline evidence
can be reused; no new baseline rental or full evaluation is triggered by this CLI.
