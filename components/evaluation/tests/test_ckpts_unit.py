"""plugins/ckpts (local, hf) on tiny fake checkpoint dirs and a fake huggingface_hub module (CPU, no network)."""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from evaluation import cli
from evaluation.core import spec as S
from evaluation.core.errors import EvalError
from evaluation.plugins.ckpts import hf as H
from evaluation.plugins.ckpts.local import LocalCkpt

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parents[2]
TASK = "open_fridge_door"
CAMS = [f"observation.images.{c}" for c in ("wrist", "overhead_left", "overhead_right")]
SHA = "a" * 40
TOKEN = "hf_dummy_token_value"


@pytest.fixture
def root(tmp_path):
    return tmp_path


@pytest.fixture
def out(root):
    return root / "run"


@pytest.fixture
def spec():
    return SimpleNamespace(out="run", train=SimpleNamespace(task=TASK, robot="mobile_panda"))


def fake_ckpt(d, keys=CAMS, shape=(3, 480, 480), rename_map=None):
    d.mkdir(parents=True, exist_ok=True)
    feats = {"observation.state": dict(type="STATE", shape=[6])}
    feats.update({k: dict(type="VISUAL", shape=list(shape)) for k in keys})
    (d / "config.json").write_text(json.dumps(dict(type="smolvla", input_features=feats,
                                                   output_features={"action": dict(type="ACTION", shape=[11])})))
    (d / "model.safetensors").write_bytes(b"weights")
    (d / "policy_preprocessor.json").write_text("{}")
    if rename_map is not None:
        (d / "train_config.json").write_text(json.dumps(dict(rename_map=rename_map)))
    return d


# ------------------------------------------------------------------ local
def test_local_resolves_a_ckpt_under_runs(spec, root, out):
    fake_ckpt(root / "runs" / "ckpts" / "c")
    r = LocalCkpt(spec, {"path": "runs/ckpts/c"}).resolve(out)
    assert r.path == str(root / "runs" / "ckpts" / "c")
    assert r.origin["source"] == "local" and r.origin["path"] == "runs/ckpts/c" and r.origin["model_bytes"] == 7
    assert len(r.origin["config_sha256"]) == 64


def test_local_refuses_a_path_outside_runs(spec, root, out):
    fake_ckpt(root / "elsewhere")
    with pytest.raises(EvalError, match="not under"):
        LocalCkpt(spec, {"path": "elsewhere"}).resolve(out)
    with pytest.raises(EvalError, match="not under"):
        LocalCkpt(spec, {"path": "runs/../elsewhere"}).resolve(out)


def test_local_needs_config_json_and_a_path(spec, root, out):
    (root / "runs" / "ckpts" / "empty").mkdir(parents=True)
    with pytest.raises(EvalError, match="no config.json"):
        LocalCkpt(spec, {"path": "runs/ckpts/empty"}).resolve(out)
    with pytest.raises(EvalError, match="needs `path`"):
        LocalCkpt(spec, {})


# ------------------------------------------------------------------ hf
@pytest.fixture
def hub(monkeypatch, root):
    """A fake huggingface_hub: records calls, 'downloads' a ckpt into local_dir[/subfolder]."""
    calls = []

    def snapshot_download(repo_id, **kw):
        calls.append(("download", repo_id, kw))
        sub = kw["allow_patterns"][0].removesuffix("/*") if kw.get("allow_patterns") else ""
        fake_ckpt(Path(kw["local_dir"]) / sub)
        return str(kw["local_dir"])

    class HfApi:
        def model_info(self, repo, revision=None, **kw):
            calls.append(("info", repo, revision, kw))
            return SimpleNamespace(sha=SHA)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download, HfApi=HfApi))
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    return calls


def test_hf_downloads_the_subfolder_into_the_commit_cache(hub, spec, root, out):
    r = H.HfCkpt(spec, {"repo": "org/name", "subfolder": "checkpoints/030000/pretrained_model"}).resolve(out)
    cache = root / "runs" / "ckpt_cache" / "hf" / "org__name" / SHA
    assert r.path == str(cache / "checkpoints/030000/pretrained_model")
    assert hub[0] == ("info", "org/name", "main", {})
    kind, repo, kw = hub[1]
    assert (repo, kw) == ("org/name", dict(revision=SHA, local_dir=cache, allow_patterns=["checkpoints/030000/pretrained_model/*"]))
    assert r.origin["source"] == "hf" and r.origin["commit"] == SHA and r.origin["revision"] == "main"
    assert r.origin["subfolder"] == "checkpoints/030000/pretrained_model"
    assert r.origin["path"] == str(Path(r.path).relative_to(root))


def test_hf_whole_repo_and_pinned_commit_make_no_info_call(hub, spec, out):
    r = H.HfCkpt(spec, {"repo": "org/name", "revision": SHA}).resolve(out)
    assert [c[0] for c in hub] == ["download"] and hub[0][2]["allow_patterns"] is None
    assert r.origin["subfolder"] is None and r.origin["commit"] == SHA


def test_hf_reuses_a_pinned_download_without_network(hub, spec, out):
    plugin = H.HfCkpt(spec, {"repo": "org/name", "revision": SHA, "subfolder": "a"})
    first = plugin.resolve(out)
    hub.clear()
    assert plugin.resolve(out) == first and hub == []


def test_hf_moving_revision_checks_the_commit_but_does_not_redownload(hub, spec, out):
    plugin = H.HfCkpt(spec, {"repo": "org/name"})
    plugin.resolve(out)
    hub.clear()
    plugin.resolve(out)
    assert [c[0] for c in hub] == ["info"]


def test_hf_never_passes_or_writes_the_token(hub, spec, root, out):
    r = H.HfCkpt(spec, {"repo": "org/name"}).resolve(out)
    assert all("token" not in (c[2] if c[0] == "download" else c[3]) for c in hub)
    assert TOKEN not in json.dumps(r.origin)
    assert not [p for p in root.rglob("*") if p.is_file() and TOKEN.encode() in p.read_bytes()]


def test_hf_a_repo_without_config_json_fails(hub, spec, out, monkeypatch):
    monkeypatch.setattr(sys.modules["huggingface_hub"], "snapshot_download", lambda repo_id, **kw: None)
    with pytest.raises(EvalError, match="no config.json"):
        H.HfCkpt(spec, {"repo": "org/name", "revision": SHA}).resolve(out)


@pytest.mark.parametrize("params", [{}, {"repo": "name"}, {"repo": "a/b/c"}, {"repo": "org/name", "revision": "../x"},
                                    {"repo": "org/name", "subfolder": "../x"}, {"repo": "org/name", "token": "x"}])
def test_hf_rejects_invalid_params(spec, params):
    with pytest.raises(EvalError):
        H.HfCkpt(spec, params)


def test_importing_the_component_does_not_load_huggingface_hub():
    code = ("import sys; from evaluation import cli; from evaluation.core import pipeline; "
            "import evaluation.plugins.ckpts.hf, evaluation.plugins.ckpts.local; "
            "assert 'huggingface_hub' not in sys.modules")
    paths = [REPO / "components/evaluation/src", REPO / "components/datagen/src", REPO / "common"]
    subprocess.run([sys.executable, "-c", code], check=True, env={**os.environ, "PYTHONPATH": os.pathsep.join(map(str, paths))})


def test_the_cli_builds_the_ckpt_source_of_the_spec(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(TESTS))
    (tmp_path / "train.yaml").write_text(yaml.safe_dump({
        "task": "t", "robot": "r", "out": "train_run", "generator": {"plugin": "fake_datagen"},
        "randomisers": {"fake_datagen": {"x": [-0.5, 0.5]}}, "splits": {"train": {"target": 3, "pool": {"initial": [0, 9]}}}}))
    raw = {"train": "train.yaml", "out": "run", "ckpt": {"source": "local", "path": "runs/c"}, "policy": {"plugin": "fake_eval"},
           "rollout": {"plugin": "fake_eval"}, "judge": {"criterion": "fake_eval"},
           "sets": {"heldout": {"plugin": "fake_eval", "count": 1, "pool": {"initial": [100, 109]}}}}
    wrapped = cli.build_plugins(S.parse(raw, root=tmp_path), ("rollout",)).ckpt
    assert isinstance(wrapped, LocalCkpt)
    plain = cli.build_plugins(S.parse(raw | {"ckpt": {"source": "fake_eval"}}, root=tmp_path), ("rollout",)).ckpt
    assert type(plain).__name__ == "FakeCkpt"
