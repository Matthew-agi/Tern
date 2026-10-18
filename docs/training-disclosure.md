# Training and evaluation disclosure

The owner has selected Apache-2.0 for Tern. This document preserves the training and evaluation evidence accompanying that choice; source dataset terms and attribution remain separate. The serving-repo implementation does not change Hub visibility or submit the benchmark.

Base: Qwen/Qwen3.6-35B-A3B at `995ad96eacd98c81ed38be0c5b274b04031597b0`. Submitted candidate: final optimizer checkpoint 1674, approximately 500M consumed tokens. Frozen INT8 expert backbone during adaptation, selected expert LoRA rank 2, shared LoRA rank 16, jointly trained decision head; serving merges adapters into the training-rounded BF16 backbone. Full depth (40 layers), eight routed experts, no early exit.

Training sources in the deployment mixture manifest:

ag_news, agent_tools, anli, babi, banking77, beavertails, bigbench, blimp, boolq, breaking_nli, cladder_generated, clinc, codeio, contract_nli, corr2cause, cybersecurity, esci, financial_phrasebank, glue, go_emotions, gsm8k, habermas, hallucination, hans, helpsteer3, hotpotqa, hover, jailbreak, jev_distill, lex_glue, logiqa2, massive, math, medmcqa, mnli, msmarco, nemotron_pivot, open_jev, paws, proofwriter, quality, ragtruth, reclor, shp, super_glue, swag, toxigen, trec, wanli, when2call, xcode, yelp.

The mixture includes teacher-derived `jev_distill` and `open_jev` data. Do not claim this is teacher-free or that source-family overlap implies test-row overlap.

Validation during training included 150 Decision Index-derived BBH, MMLU-Pro and When2Call requests, plus mixture validation. Those public validation results were observed during development. Earlier checkpoints were also used for public-suite evaluations and expert/early-exit investigations; the submitted configuration here retains all eight experts and all layers.

The final audit scanned all **428,507 consumed records / 499,999,909 tokens**, through checkpoint 1674's saved cursor, against all **140,178 public evaluation requests**. Recorded native splits were `train`, except BeaverTails' `330k_train`. Split metadata does not prove independence from benchmark inputs derived from other sources.

The strict full-request/state audit found zero matches. Full requests ignored question IDs and option order; the state checks also included Unicode/case/whitespace normalization. Different prompt wrappers make that audit insufficient by itself.

A second audit searched normalized evaluation text as word-bounded substrings in training states and instructions. It found **3,238 candidate requests with shared input/context text**. It checked state text leaves of at least 80 characters / 12 words, and complete BANKING77/CLINC intent utterances of at least 12 characters / 3 words. Text appearing in more than 10 evaluation requests was excluded as common context. These are candidate text/context overlaps, **not 3,238 proven identical labeled test decisions**.

| Public benchmark | Candidate requests | Interpretation boundary |
|---|---:|---|
| RouterBench (auxiliary; outside the index) | 1,846 | Shared underlying prompts, including GSM8K training problems |
| HoVer | 668 | Shared state text, potentially evidence passages rather than the same claim |
| SATA-Bench | 321 | Shared passages do not alone establish the same decision/questions |
| CLINC150/OOS | 187 | Complete normalized user utterance occurs in training input |
| BANKING77 | 106 | Complete normalized user utterance occurs in training input |
| MMLU-Pro | 48 | Shared state text; potentially underlying problem overlap |
| Amazon ESCI | 32 | Shared state text may be product context rather than identical relevance decisions |
| ToolRet | 26 | Shared state text may be tool/context descriptions |
| BRIGHT | 3 | Shared state text may be document context |
| ContractNLI | 1 | Shared contract text does not alone establish the same labeled hypotheses |

Of these candidates, **1,392 are in index benchmarks** and 1,846 are auxiliary RouterBench rows. The audit retains affected run IDs, training source/split/native IDs, and matched-text hashes, without redistributing benchmark or training text. Original predictions and scores are unchanged; no post-hoc removal or altered response is presented as the submitted run. Maintainers can inspect the disclosure and apply their own overlap policy.

This is not a semantic paraphrase/near-duplicate audit. It does not establish the absence of partial overlaps, base-model pretraining exposure, or teacher knowledge. Public validation exposure and development evaluations remain disclosed above.

## Licensing review evidence

The consumed metadata contains **8,678 ANLI** and **4,304 ContractNLI** records tagged `license_use: non-commercial`. These are intermediate dataset tags, not a conclusion about the trained weights.

- The original [ANLI repository](https://github.com/facebookresearch/anli#license) declares a Creative Commons non-commercial license.
- The original [ContractNLI publisher](https://stanfordnlp.github.io/contract-nli/#license) declares **CC BY 4.0**, which conflicts with the intermediate tag and needs reconciliation against the exact source variant.
- [Tasksource's documentation](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions/blob/f5f61df425637f15fa8ef4ce26d7dcbf9f4be0ce/README.md) describes the tags as a best-effort combination of source-card and provenance metadata; original source terms still apply.

Many other consumed sources have no `license_use` tag in this training export. That does not mean they have no restrictions or have been cleared. The current review is not a complete legal clearance of all 52 consumed source families.

Training file SHA256: `735a2276bdcc49152b3d91e9a2c46ac59763056fdc4c920a1d024c9473ffd499`.
Validation file SHA256: `5f1340a06208b067e6203b429e48fec5c099a0d2dfe12350c511af210a9cb2e7`.
