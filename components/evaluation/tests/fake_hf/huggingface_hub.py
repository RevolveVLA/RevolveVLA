"""A stand-in for huggingface_hub (put its dir first on $EVAL_PLUGIN_PATH): no network, a tiny checkpoint written locally.
Calls are appended to $FAKE_HF_LOG as `model_info <repo> <revision>` / `snapshot_download <repo> <revision> <patterns>`."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

SHA = "0123456789abcdef0123456789abcdef01234567"


def logged(line):
    with open(os.environ["FAKE_HF_LOG"], "a") as f:
        f.write(line + "\n")


class HfApi:
    def model_info(self, repo, revision=None):
        logged(f"model_info {repo} {revision}")
        return SimpleNamespace(sha=SHA)


def snapshot_download(repo, revision=None, local_dir=None, allow_patterns=None, **kw):
    logged(f"snapshot_download {repo} {revision} {allow_patterns}")
    sub = Path(local_dir) / (allow_patterns[0].rsplit("/*", 1)[0] if allow_patterns else "")
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "config.json").write_text(json.dumps({"input_features": {}}))
    (sub / "model.safetensors").write_bytes(b"\0" * 16)
    return str(local_dir)
