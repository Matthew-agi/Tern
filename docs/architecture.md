# Runtime architecture

`tern serve` owns two processes: a loopback-only vLLM pooling backend and a
SystemOne HTTP frontend. It selects one idle GPU, verifies installed vLLM source
and all model hashes, starts both processes, then checks a real inference before
reporting ready. Interrupts or a failed child shut down only its own process
groups. It never stops unrelated GPU work. A second GPU occupancy check runs
after the potentially long model download.

The package registers `tern_pooler = kev_vllm:register` as a vLLM general plugin.
That plugin installs the frozen metadata transport and registers
`ClefQwenMoeForPooling`. Encoding produces the exact original token sequence and
question/option spans. The causal backbone processes the full input through all
40 layers; the joint schema head uses hidden states, schema spans and the output
embedding to score every supplied option. The head is part of the pooling model,
not a separate CPU classifier or a generative prompt.

The merged checkpoint already includes backbone adapters and
`clef_head.safetensors`. The trained head is stored in FP32; its original runtime
precasts selected parameters to BF16 and uses BF16 autocast. The output embedding
is retained because the trained head uses it. Changing it, the prompt, option
ordering, head pooling, normalization or precision would change model behavior.
No such refactoring was made here.

`src/serve_clef_api.py` and the entire copied numerical runtime remain byte-identical
to the final packaged step-1674 runtime. `docs/frozen-runtime-sha256.json` records
their hashes. A small wrapper sets the public model name to `tern` and adds a
readiness endpoint; it does not rewrite probabilities or requests. Historical
`kev_vllm`/CLEF names are intentional provenance and compatibility details.

Endpoints:

- `POST /v1/systemone`: one JSON state plus named typed questions.
- `POST /v1/systemone/batch`: `{"requests": [...]}`, 1–256 complete records.
- `GET /health`: frontend and backend liveness.
- `GET /ready`: liveness after startup inference has completed.

Question types are `choice` (named criteria), `score` (ordered criteria) and
`noul` (proposition probability). Inputs exceeding 32,768 encoded tokens across
the complete state and schema are rejected, never silently truncated. This
release is text-only, despite unused multimodal helpers in the frozen source.

Stateless prefill removes resident generation/recurrent caches because every
request is processed as a complete prompt. It retains causal attention and all
recurrent arithmetic. Prefix caching, chunked prefill, async scheduling and
generation are intentionally unavailable under this contract. Piecewise graphs
are enabled; eager mode is not used.
