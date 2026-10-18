from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from tern import launch
from tern.config import profile


@contextmanager
def health_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/health"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_empty_vllm_health_body_is_ready():
    with health_server() as url:
        launch.wait_health(url, [SimpleNamespace(poll=lambda: None)], 1)


def test_early_backend_exit_is_not_ready():
    with pytest.raises(RuntimeError, match="exited"):
        launch.wait_health("http://127.0.0.1:1/health", [SimpleNamespace(poll=lambda: 1)], 1)


def test_busy_selected_gpu_is_rejected_without_stopping_anything(monkeypatch):
    calls = []
    def output(cmd, **kwargs):
        calls.append(cmd)
        if "--query-compute-apps=pid" in cmd:
            return "12345\n"
        return "NVIDIA A100-SXM4-80GB, GPU-chosen, 81920, 580.0\n"
    monkeypatch.setattr(launch.subprocess, "check_output", output)
    with pytest.raises(RuntimeError, match="already has compute"):
        launch.inspect_gpu("3", profile("a100"))
    assert calls[0][2] == "3" and calls[1][2] == "GPU-chosen"


@pytest.mark.parametrize("name,memory", [("NVIDIA RTX 6000 Ada", 49152), ("NVIDIA RTX PRO 6000", 49152)])
def test_wrong_gpu_or_memory_is_rejected(monkeypatch, name, memory):
    monkeypatch.setattr(launch.subprocess, "check_output", lambda *a, **k: f"{name}, GPU-x, {memory}, 580.0\n")
    with pytest.raises(RuntimeError, match="needs"):
        launch.inspect_gpu("0", profile("rtx-pro-6000"))


def test_port_checks_reject_existing_listener():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        with pytest.raises(OSError):
            launch.check_ports("127.0.0.1", sock.getsockname()[1], 1)
    with pytest.raises(ValueError):
        launch.check_ports("127.0.0.1", 8101, 8101)


def test_cleanup_does_not_kill_unrelated_process():
    command = [sys.executable, "-c", "import time; time.sleep(120)"]
    owned = subprocess.Popen(command, start_new_session=True)
    unrelated = subprocess.Popen(command, start_new_session=True)
    try:
        launch.stop_processes([owned])
        assert owned.poll() is not None
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_frontend_failure_cleans_up_backend(monkeypatch, tmp_path):
    from tern import preflight
    monkeypatch.setattr(launch, "inspect_gpu", lambda *a: {"uuid": "GPU-test"})
    monkeypatch.setattr(launch, "check_ports", lambda *a: None)
    monkeypatch.setattr(preflight, "check_runtime", lambda: None)
    monkeypatch.setattr(launch, "download", lambda *a: tmp_path)
    monkeypatch.setattr(launch, "wait_health", lambda *a: None)
    original = subprocess.Popen
    children = []
    def start(cmd, **kwargs):
        if children:
            raise OSError("synthetic frontend failure")
        child = original([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
        children.append(child)
        return child
    monkeypatch.setattr(launch.subprocess, "Popen", start)
    args = SimpleNamespace(profile="a100", gpu="0", host="127.0.0.1", port=8000,
                           backend_port=8101, model=None, cache_dir=tmp_path, startup_timeout=1)
    with pytest.raises(OSError, match="synthetic"):
        launch.serve(args)
    assert children[0].poll() is not None
