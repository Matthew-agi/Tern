# Packaged H100 deployment — October 22, 2026

A fresh clone at `ec475b598306ebe29ea019533e6c6f78821d6674` completed all five configurations below through the documented Docker/CLI path on one **H100 SXM 80GB HBM3 (700W)**. That tested serving commit is now on `main`. No serving-code edits, manual checkpoint conversion, dependency overrides or eager-mode fallback were needed.

| Configuration | Mean input tok/s | GPU $/M input tokens at $2.35/hour |
|---|---:|---:|
| Original 8/8, 40 layers | 80,173 | $0.00814 |
| Original pruned 4/8, 40 layers | 91,875 | $0.00711 |
| Original pruned 4/8, exit 30 | 121,267 | $0.00538 |
| Trained 4/8, 40 layers | 95,244 | $0.00685 |
| Trained 4/8, exit 30 | 115,498 | $0.00565 |

The matched panel contains 257 requests / 580,694 encoded input tokens, at concurrency 256. Each configuration received one complete-panel warmup and three timed repetitions; all repetitions were retained, including slower first repetitions for original 4/8 and trained 4/8 exit-30. These are short concurrent bursts including HTTP, encoding, inference and response handling. They exclude provisioning, downloads, adapter merge, startup, compilation warmup and idle time. $2.35/hour is an explicit comparison assumption, not a market quote or the rental invoice. Cost = hourly rate × 1,000,000 / (3,600 × mean input tok/s). At lower productive utilization, effective cost increases accordingly.

Historical [quality and performance results](execution-policies.md) used H100 PCIe hardware. The SXM throughput improvement cannot be attributed solely to software. This deployment test did not rerun quality evaluation; historical public-sample scores are not new SXM measurements.

## Integrity and numerical limits

All responses passed the packaged validator and matched the required token count. Two serial repetitions on 32 matched requests / 195 decisions were identical for every configuration. Concurrent grouping still changed probabilities: maximum adjacent-run absolute differences ranged from 0.033 to 0.213 across configurations, with up to 12 changed argmax decisions out of 2,088 in one comparison. Successful deployment does not establish cross-batch numerical equivalence or a new accuracy score.

The automatic continuation merge consumed 522 trainable tensors. A separate CPU audit verified all 122 indexed `pooler.head.*` tensors exactly against the pinned continuation checkpoint. The trained continuation and original model use separate caches. No numerical tolerance was widened.

## Host setup and reproducibility

- Host: Ubuntu 22.04.5, NVIDIA driver 570.148.08, Docker 28.3.1 and Compose 2.38.1.
- Container runtime: Torch 2.13.0+cu129, vLLM 0.30.0+cu129, Triton 3.7.1 and FlashInfer 0.6.18.post1.
- The provider had NVIDIA Container Toolkit installed but not registered with Docker. Registration with `sudo nvidia-ctk runtime configure --runtime=docker`, followed by restarting the idle Docker daemon, was required. Follow NVIDIA's host setup instructions before launching GPU containers.
- The host user required `sudo` for Docker. Pass profile variables through `sudo env` on such hosts; no Docker socket permissions were changed.
- Original weights are public. The separate `top4-50m` continuation requires an authorized HF read token; it downloaded and merged automatically.
- Packaged MoE tile filenames target H100 PCIe. On SXM, vLLM reported missing matching tuning files and used defaults. No filenames were changed to mask that fallback; SXM-specific tile tuning remains untested.
- Readiness uses a small synthetic request. Warm the complete representative panel before timing, because larger shapes can trigger additional compilation.
- The frozen tokenizer's existing regex warning was retained; all measured panels matched the historical encoded token count.

These results cover original 8/8, original 4/8 and trained 4/8 at the listed depths. Other policies and A100/RTX combinations require their own GPU validation.
