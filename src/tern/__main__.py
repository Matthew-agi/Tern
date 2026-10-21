import argparse
import json
import os
from pathlib import Path

from .config import PROFILES, backend_command, profile, release


def execution_arguments(parser):
    parser.add_argument("--experts", type=int, default=os.environ.get("TERN_EXPERTS") or None,
                        help="Dynamic routed experts per token, 1-8 (default: 8; adapter: 4)")
    parser.add_argument("--normalization-reference", type=int,
                        default=os.environ.get("TERN_NORMALIZATION_REFERENCE") or 8,
                        help="Retain top-N probability mass as denominator (default: 8)")
    parser.add_argument("--exit-layer", type=int, default=os.environ.get("TERN_EXIT_LAYER") or 40,
                        help="Unconditionally execute the first N layers, 1-40 (default: 40)")
    parser.add_argument("--adapter", choices=("top4-50m",),
                        default=os.environ.get("TERN_ADAPTER") or None,
                        help="Merge the pinned private continuation and its trained head")


def main():
    parser = argparse.ArgumentParser(description="Tern typed-decision serving")
    commands = parser.add_subparsers(dest="command", required=True)
    server = commands.add_parser("serve", help="Download, verify, warm up and serve")
    server.add_argument("--profile", choices=PROFILES, default="rtx-pro-6000")
    server.add_argument("--model", type=Path, help="Use a local copy of the exact release")
    server.add_argument("--cache-dir", default=os.environ.get("TERN_CACHE_DIR", "~/.cache/tern"))
    server.add_argument("--gpu", default="0", help="nvidia-smi index or GPU UUID")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8000)
    server.add_argument("--backend-port", type=int, default=8101)
    server.add_argument("--startup-timeout", type=int, default=1800)
    plan = commands.add_parser("plan", help="Print the exact configuration without loading a GPU")
    plan.add_argument("--profile", choices=PROFILES, default="rtx-pro-6000")
    fetch = commands.add_parser("download", help="Download and SHA-256 verify the frozen release")
    fetch.add_argument("--cache-dir", default=os.environ.get("TERN_CACHE_DIR", "~/.cache/tern"))
    fetch.add_argument("--model", type=Path)
    commands.add_parser("check-runtime", help="Check installed vLLM source against frozen patch contracts")
    bench = commands.add_parser("benchmark", help="Measure serial or concurrent HTTP inference")
    bench.add_argument("--base-url", default="http://127.0.0.1:8000")
    bench.add_argument("--input", type=Path, required=True, help="JSONL requests; never uploaded")
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--concurrency", type=int, default=1)
    bench.add_argument("--batch-size", type=int, default=1, help="Records per HTTP request, 1 to 256")
    bench.add_argument("--warmup", type=int, default=8)
    compare = commands.add_parser("compare", help="Compare paired benchmark response files")
    compare.add_argument("reference", type=Path)
    compare.add_argument("candidate", type=Path)
    for command in (server, plan):
        execution_arguments(command)
    fetch.add_argument("--adapter", choices=("top4-50m",), help="Pre-merge the private continuation")
    args = parser.parse_args()
    if args.command in ("serve", "plan"):
        from .policy import ExecutionPolicy
        try:
            policy = ExecutionPolicy.from_args(args)
            if args.adapter not in (None, "top4-50m"):
                raise ValueError("Unknown adapter")
        except ValueError as exc:
            parser.error(str(exc))
    if args.command == "serve":
        from .launch import serve
        serve(args)
    elif args.command == "plan":
        p = profile(args.profile)
        from .adapters import adapter_spec, artifact_key
        from .config import worker_environment
        cache = Path("CACHE") / p["id"] / artifact_key(args.adapter) / policy.cache_key
        env = worker_environment(p, Path("MODEL_SNAPSHOT"), cache, 8101, "tern", "0", policy)
        print(json.dumps({"profile": p, "policy": policy.as_dict(),
                          "model": {k: v for k, v in release().items() if k != "files"},
                          "adapter": adapter_spec(args.adapter) if args.adapter else None,
                          "cache": str(cache),
                          "environment": {k: v for k, v in env.items() if k.startswith(("CLEF_", "VLLM_", "TERN_H100_")) or k == "CUBLAS_WORKSPACE_CONFIG"},
                          "command": backend_command(p, "MODEL_SNAPSHOT", 8101, "tern", policy)}, indent=2))
    elif args.command == "download":
        from .artifacts import download
        model = download(args.cache_dir, args.model)
        if args.adapter:
            from .adapters import prepare_adapter
            model = prepare_adapter(model, args.cache_dir, args.adapter)
        print(model)
    elif args.command == "check-runtime":
        from .preflight import check_runtime
        check_runtime()
        print("Pinned vLLM source contract matches")
    elif args.command == "benchmark":
        from .benchmark import benchmark
        benchmark(args)
    elif args.command == "compare":
        from .benchmark import compare
        print(json.dumps(compare(args.reference, args.candidate), indent=2))


if __name__ == "__main__":
    main()
