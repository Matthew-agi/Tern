import argparse
import json
import os
from pathlib import Path

from .config import PROFILES, backend_command, profile, release


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
    args = parser.parse_args()
    if args.command == "serve":
        from .launch import serve
        serve(args)
    elif args.command == "plan":
        p = profile(args.profile)
        print(json.dumps({"profile": p, "model": {k: v for k, v in release().items() if k != "files"},
                          "command": backend_command(p, "MODEL_SNAPSHOT", 8101, "tern")}, indent=2))
    elif args.command == "download":
        from .artifacts import download
        print(download(args.cache_dir, args.model))
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
