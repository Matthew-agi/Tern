"""Fetch only the exact, hash-verified inference artifacts; never fetch training data."""
import hashlib
import json
from pathlib import Path
import shutil

from .config import release


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(model, manifest=None):
    manifest = manifest or release()
    for name, expected in manifest["files"].items():
        path = model / name
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Missing or wrong-size model file: {path}")
        if sha256(path) != expected["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")
    config = json.loads((model / "config.json").read_text())
    export = json.loads((model / "export.json").read_text())
    if (export["checkpoint_step"] != manifest["checkpoint_step"] or
            config["num_hidden_layers"] != 40 or config["num_experts_per_tok"] != 8 or
            config["architectures"] != [manifest["architecture"]]):
        raise ValueError("This release requires the frozen full-depth, top-8 checkpoint")
    print(f"Verified {len(manifest['files'])} inference files", flush=True)
    return model


def download(cache, local=None):
    if local is not None:
        return verify(Path(local).expanduser().resolve())
    from huggingface_hub import snapshot_download
    manifest = release()
    hub = Path(cache).expanduser().resolve() / "huggingface"
    hub.mkdir(parents=True, exist_ok=True)
    kwargs = dict(repo_id=manifest["repository"], revision=manifest["revision"],
                  cache_dir=str(hub), allow_patterns=list(manifest["files"]))
    # Check existing snapshots before requiring space for another 70 GB.
    try:
        path = Path(snapshot_download(**kwargs, local_files_only=True))
        if all((path / name).is_file() for name in manifest["files"]):
            return verify(path)
    except (FileNotFoundError, OSError):
        pass
    required = sum(f["bytes"] for f in manifest["files"].values()) + 10 * 1024**3
    if shutil.disk_usage(hub).free < required:
        raise RuntimeError("Allow at least 80 GB free for the model download, plus container/compiler storage")
    print(f"Downloading {manifest['repository']} at {manifest['revision']}", flush=True)
    return verify(Path(snapshot_download(**kwargs)))
