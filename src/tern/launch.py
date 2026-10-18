"""Own the complete server lifecycle without touching unrelated GPU processes."""
import csv
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

from .artifacts import download
from .config import backend_command, profile, release, worker_environment

SMOKE = {"model": "tern", "state": "The invoice status is paid.", "questions": {
    "status": {"type": "choice", "instructions": "What is the invoice status?",
               "criteria": {"paid": "Paid", "unpaid": "Unpaid"}}}}


def inspect_gpu(selector, p):
    raw = subprocess.check_output(["nvidia-smi", "-i", str(selector),
        "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv,noheader,nounits"], text=True)
    rows = list(csv.reader(io.StringIO(raw)))
    if len(rows) != 1:
        raise RuntimeError("Select exactly one GPU with --gpu INDEX or UUID")
    name, uuid, memory, driver = [x.strip() for x in rows[0]]
    if p["gpu_contains"] not in name or float(memory) < p["minimum_memory_mib"]:
        raise RuntimeError(f"Profile {p['id']} needs {p['gpu_contains']} with sufficient VRAM; found {name}, {memory} MiB")
    busy = subprocess.check_output(["nvidia-smi", "-i", uuid,
        "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if busy:
        raise RuntimeError(f"Selected GPU already has compute processes: {busy}; no process was stopped")
    return dict(name=name, uuid=uuid, memory_mib=float(memory), driver=driver)


def check_ports(host, port, backend_port):
    if port == backend_port:
        raise ValueError("Public and internal ports must differ")
    for address, number in [(host, port), ("127.0.0.1", backend_port)]:
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((address, number))


def request(url, body=None, timeout=5):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def wait_health(url, processes, timeout):
    deadline = time.monotonic() + timeout
    next_message = time.monotonic() + 30
    while time.monotonic() < deadline:
        if any(p.poll() is not None for p in processes):
            raise RuntimeError("A server process exited during startup; see its log above")
        try:
            # vLLM's health endpoint returns an empty 200 body.
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except (OSError, ValueError):
            pass
        if time.monotonic() >= next_message:
            print("Waiting for model loading / compilation...", flush=True)
            next_message = time.monotonic() + 30
        time.sleep(1)
    raise TimeoutError(f"Startup exceeded {timeout}s; see logs, or increase --startup-timeout")


def stop_processes(processes):
    # Each child owns a new session, including any vLLM grandchildren.
    for process in reversed(processes):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 30
    for process in reversed(processes):
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    # The leader can exit before its GPU worker; clean up its remaining group too.
    for process in reversed(processes):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def serve(args):
    p = profile(args.profile)
    print(p["validation"], flush=True)
    gpu = inspect_gpu(args.gpu, p)
    check_ports(args.host, args.port, args.backend_port)
    from .preflight import check_runtime
    check_runtime()
    model = download(args.cache_dir, args.model)
    # Recheck after a potentially long download before starting GPU work.
    inspect_gpu(gpu["uuid"], p)
    processes = []
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        with tempfile.TemporaryDirectory(prefix="tern-") as temp:
            ready = Path(temp) / "ready"
            cache = Path(args.cache_dir).expanduser().resolve() / p["id"]
            cache.mkdir(parents=True, exist_ok=True)
            env = worker_environment(p, model, cache, args.backend_port, "tern", gpu["uuid"])
            env["TERN_READY_FILE"] = str(ready)
            command = backend_command(p, model, args.backend_port, "tern")
            print(json.dumps({"profile": p, "gpu": gpu, "release_revision": release()["revision"],
                              "backend_command": command}), flush=True)
            processes.append(subprocess.Popen(command, env=env, start_new_session=True))
            wait_health(f"http://127.0.0.1:{args.backend_port}/health", processes, args.startup_timeout)
            front = [sys.executable, "-m", "uvicorn", "tern.api:app", "--host", args.host,
                     "--port", str(args.port), "--no-access-log"]
            processes.append(subprocess.Popen(front, env=env, start_new_session=True))
            host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
            base = f"http://{host}:{args.port}"
            wait_health(base + "/health", processes, args.startup_timeout)
            from .validation import validate_response
            print("Warming up with a synthetic request...", flush=True)
            validate_response(SMOKE, request(base + "/v1/systemone", SMOKE, timeout=args.startup_timeout))
            ready.touch()
            print(f"Tern READY: {base}/v1/systemone (profile={p['id']})", flush=True)
            while all(process.poll() is None for process in processes):
                time.sleep(1)
            ready.unlink(missing_ok=True)
            raise RuntimeError("A server process exited; shutting down its companion")
    finally:
        # Ignore repeated stop requests while the owned process groups exit.
        for s in previous:
            signal.signal(s, signal.SIG_IGN)
        stop_processes(processes)
        for s, handler in previous.items():
            signal.signal(s, handler)
