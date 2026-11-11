"""Unit tests (no docker, no GPU, no LeRobot): progress parser, retrain-mode plans, checkpoint finalize, and the
gpu_machine / modal backends against a fake ssh/rsync runner and a fake `modal` module."""
import json
import os
import shutil
import subprocess
import types
from pathlib import Path

import pytest

import gpu_machine as G
import modal_backend as M
import progress as P
import train as T

LOG = ("INFO 2026-11-09 00:14:27 ot_train.py:641 step:1K smpl:2K ep:4 epch:2.21 loss:0.512 grdn:3.200 lr:1.0e-04 "
       "updt_s:0.140 data_s:0.012 smp/s:13 mem_gb:1.86")
BAR = "Training:  15%|##        | 3/20 [00:05<00:30,  1.52s/step]"


def test_progress_log_bar_and_counted_steps():
    P.reset()
    assert P.parse("PROGRESS {\"log_freq\": 5, \"phase\": \"train\"}") == {"log_freq": 5, "phase": "train"}
    d = P.parse(LOG)
    assert d["step"] == 5 and d["loss"] == 0.512 and d["lr"] == 1e-4 and d["grad_norm"] == 3.2
    assert d["gpu_mem_gb"] == 1.86 and d["s_per_step"] == 0.152 and d["losses"] == [[5, 0.512]]
    assert P.parse(LOG)["losses"] == [[5, 0.512], [10, 0.512]]
    assert P.parse(BAR) == dict(step=3, total_steps=20, elapsed_s=5, eta_s=30)
    assert P.parse("Training:   0%|          | 0/20 [00:00<?, ?step/s]") == dict(step=0, total_steps=20, elapsed_s=0)
    assert P.parse("random line") is None
    P.reset()
    assert P.parse(LOG)["step"] == 1000            # no log_freq reported: LeRobot's rounded number
    assert P.big("2M") == 2_000_000 and P.big("7") == 7


def test_progress_serve():
    assert P.parse_serve("INFO ... RosettaPolicyServer started on 0.0.0.0:8080") == dict(ready=True, address="0.0.0.0:8080")
    assert P.parse_serve("INFO Action chunk #0 generated | Total time: 580ms") == dict(chunks=1)


def ckpt(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for f in T.CKPT_FILES:
        (d / f).write_text("x")
    return d


def test_plan_modes(tmp_path):
    prev = ckpt(tmp_path / "prev")
    kw = dict(dataset="/d", out=str(tmp_path / "ckpt"), base="lerobot/smolvla_base", steps=40, batch_size=2)
    fb = T.plan(mode="from_base", prev=str(prev), **kw)
    assert fb["init"] == "lerobot/smolvla_base" and fb["episodes"] is None
    fl = T.plan(mode="finetune_last", prev=str(prev), **kw)
    assert fl["init"] == str(prev) and fl["args"]["policy.pretrained_path"] == str(prev)
    assert T.plan(mode="finetune_last", prev=None, **kw)["init"] == "lerobot/smolvla_base"   # turn 0
    ro = T.plan(mode="finetune_recovery_only", prev=str(prev), episodes=[1, 3], **kw)
    assert ro["init"] == str(prev) and ro["args"]["dataset.episodes"] == "[1,3]"
    a = T.argv(fl)
    assert a[0] == "lerobot-train" and "--policy.type=smolvla" in a and "--policy.push_to_hub=false" in a
    assert "--wandb.enable=false" in a and "--steps=40" in a and "--batch_size=2" in a and "--save_freq=40" in a
    assert f"--output_dir={tmp_path}/ckpt.partial/lerobot" in a and "--dataset.repo_id=revolve/dataset" in a
    with pytest.raises(ValueError):
        T.plan(mode="finetune_last", prev=str(tmp_path / "missing"), **kw)
    with pytest.raises(ValueError):
        T.plan(mode="bogus", **kw)


def test_recovery_selection_uses_dataset_labels():
    labels = T.dataset_labels()
    rows = [{"episode_index": 0, "labels/source": "demo"}, {"episode_index": 1, "labels/source": "recovery"},
            {"episode_index": 2, "labels/source": "recovery"}]
    assert labels.select(rows, source="recovery") == [1, 2]


def test_finalize_moves_last_step(tmp_path):
    p = T.plan(mode="from_base", dataset="/d", out=str(tmp_path / "ckpt"), base="b", steps=4, batch_size=1)
    for s in ("000002", "000004"):
        ckpt(Path(p["lerobot_out"]) / "checkpoints" / s / "pretrained_model")
    info = T.finalize(p, {"losses": [[2, 1.0], [4, 0.5]]})
    out = tmp_path / "ckpt"
    assert info["step"] == "000004" and (out / "model.safetensors").is_file() and not Path(p["out"] + ".partial").exists()
    assert json.loads((out / "train.json").read_text())["losses"] == [[2, 1.0], [4, 0.5]]


def test_save_every_copies_earlier_checkpoints_locally(tmp_path):
    p = T.plan(mode="from_base", dataset="/d", out=str(tmp_path / "ckpt"), base="b", steps=4, batch_size=1, save_every=2)
    assert p["args"]["save_freq"] == 2
    assert T.plan(mode="from_base", dataset="/d", out="/o", base="b", steps=4, batch_size=1)["args"]["save_freq"] == 4
    ck = Path(p["lerobot_out"]) / "checkpoints"
    with T.local_copier(p, every_s=3600) as c:
        assert c.saved() == []                     # nothing saved yet
        ckpt(ck / "000002" / "pretrained_model")
        (ck / "last").symlink_to("000002")
        ckpt(ck / "000004" / "pretrained_model")   # being written: `last` does not point at it yet
        c.poll()
        assert (tmp_path / "ckpt_000002" / "model.safetensors").is_file() and c.done == {"000002"}
        os.remove(ck / "last")
        (ck / "last").symlink_to("000004")
    assert not (tmp_path / "ckpt_000004").exists()  # the final step is finalize()'s
    assert T.saved_steps("garbage", ["000002"]) == [] and not T.local_copier(T.plan(
        mode="from_base", dataset="/d", out="/o", base="b", steps=4, batch_size=1)).on


class FakeRunner:
    """subprocess.run stand-in: records ssh / rsync calls; `fail` = substring of a command that exits 1."""

    def __init__(self, cleanup="CLEAN", fail=None):
        self.calls, self.cleanup, self.fail = [], cleanup, fail

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        text = " ".join(cmd)
        out = ""
        if self.fail and self.fail in text:
            return subprocess.CompletedProcess(cmd, 1, "", "boom")
        if text.endswith("echo $HOME"):
            out = "/home/u\n"
        elif "pkill" in text:
            out = self.cleanup + "\n"
        elif "out/lerobot/checkpoints |" in text:
            out = "000020\n"
        elif "readlink last" in text:              # CheckpointCopier: `last` -> 000010, then the listing
            out = "000010\n000010\nlast\n"
        return subprocess.CompletedProcess(cmd, 0, out, "")


def gm_plan(tmp_path, init_dir=True):
    prev = ckpt(tmp_path / "prev") if init_dir else None
    return T.plan(mode="finetune_last", dataset=str(tmp_path / "ds"), out=str(tmp_path / "ckpt"), base="lerobot/smolvla_base",
                  prev=str(prev) if prev else None, steps=20, batch_size=2)


def test_gpu_machine_round_trip_commands(tmp_path):
    run, streamed = FakeRunner(), []
    res = G.run(gm_plan(tmp_path), {"host": "u@h", "scratch": "~/revolve_scratch"}, "exp_turn_1",
                stream=lambda cmd, env, on_line: streamed.append(cmd) or 0, runner=run)
    s = "/home/u/revolve_scratch/exp_turn_1"
    assert res["rc"] == 0 and res["ckpt"] == tmp_path / "ckpt.partial" / "ckpt" and res["remote"]["cleanup"] == ["CLEAN"]
    rsyncs = [c for c in run.calls if c[0] == "rsync"]
    assert [c[-1] for c in rsyncs] == [f"u@h:{s}/bin/uv", f"u@h:{s}/bin/requirements.lock", f"u@h:{s}/data/dataset/",
                                       f"u@h:{s}/init/", f"{tmp_path}/ckpt.partial/ckpt/"]
    assert rsyncs[-1][-2] == f"u@h:{s}/out/lerobot/checkpoints/000020/pretrained_model/"
    env, train = streamed[0][-1], streamed[1][-1]
    assert "uv pip sync" in env and "--require-hashes" in env and f"export HOME={s}/home;" in env
    assert f"export HF_HOME={s}/cache/hf;" in env and "unset HF_TOKEN" in env
    assert f"--dataset.root={s}/data/dataset" in train and f"--policy.pretrained_path={s}/init" in train
    assert f"--output_dir={s}/out/lerobot" in train and "huggingface_hub.get_token()" in train
    clean = run.calls[-1][-1]
    assert "pkill -TERM -f '[/]home/u/revolve_scratch/exp_turn_1/'" in clean and f"rm -rf -- {s}" in clean
    assert "rmdir /home/u/revolve_scratch" in clean


def test_gpu_machine_save_every_fetches_intermediate_checkpoints(tmp_path):
    run = FakeRunner()
    p = T.plan(mode="from_base", dataset=str(tmp_path / "ds"), out=str(tmp_path / "ckpt"), base="b", steps=20,
               batch_size=2, save_every=10)
    streamed = []
    res = G.run(p, {"host": "u@h"}, "j", stream=lambda cmd, env, on_line: streamed.append(cmd) or 0, runner=run)
    s = "/home/u/revolve_scratch/j"
    assert res["rc"] == 0 and (tmp_path / "ckpt_000010").is_dir() and not (tmp_path / "ckpt_000010.partial").exists()
    rs = [c for c in run.calls if c[0] == "rsync" and "000010" in c[-2]]
    assert len(rs) == 1 and rs[0][-2:] == [f"u@h:{s}/out/lerobot/checkpoints/000010/pretrained_model/",
                                           f"{tmp_path}/ckpt_000010.partial/"]
    assert "--save_freq=10" in streamed[1][-1]


def test_gpu_machine_cleans_up_on_failure_and_cancel(tmp_path):
    run = FakeRunner()
    res = G.run(gm_plan(tmp_path, init_dir=False), {"host": "u@h"}, "j", stream=lambda *a: 1, runner=run)
    assert res["rc"] == 1 and "pkill" in run.calls[-1][-1]
    run = FakeRunner()

    def cancelled(*a):
        raise SystemExit(143)
    with pytest.raises(SystemExit):
        G.run(gm_plan(tmp_path, init_dir=False), {"host": "u@h"}, "j", stream=cancelled, runner=run)
    assert "pkill" in run.calls[-1][-1]
    run = FakeRunner(fail="data/dataset/")
    with pytest.raises(RuntimeError, match="rsync"):
        G.run(gm_plan(tmp_path), {"host": "u@h"}, "j", stream=lambda *a: 0, runner=run)
    assert "pkill" in run.calls[-1][-1]
    with pytest.raises(RuntimeError, match="left on"):
        G.run(gm_plan(tmp_path), {"host": "u@h"}, "j", stream=lambda *a: 0, runner=FakeRunner(cleanup="LEFT"))


class FakeModal:
    """The parts of the modal SDK the backend uses, recording what it does."""

    def __init__(self, rc=0):
        self.log, rc_ = [], rc
        log = self.log

        class FileEntryType:
            FILE, DIRECTORY = 1, 2

        class Entry:
            def __init__(self, path, type_):
                self.path, self.type = path, type_

        class Volume:
            files = {}

            @staticmethod
            def from_name(name, create_if_missing=False):
                log.append(("volume", name, create_if_missing))
                return Volume()

            def batch_upload(self, force=False):
                vol = self

                class B:
                    def __enter__(s):
                        return s

                    def __exit__(s, *a):
                        return False

                    def put_directory(s, src, dst):
                        log.append(("put", src, dst))
                return B()

            def reload(self):
                pass

            def listdir(self, path, recursive=False):
                base = path.rstrip("/")
                if not recursive:
                    return [Entry(f"{base}/000020", FileEntryType.DIRECTORY), Entry(f"{base}/last", FileEntryType.DIRECTORY)]
                return [Entry(f"{base}/model.safetensors", FileEntryType.FILE), Entry(f"{base}/config.json", FileEntryType.FILE)]

            def read_file_into_fileobj(self, path, f):
                f.write(b"x")

            def remove_file(self, path, recursive=False):
                log.append(("remove", path, recursive))

        class Image:
            @staticmethod
            def debian_slim(python_version):
                log.append(("image", python_version))
                return Image()

            def __getattr__(self, name):
                def step(*a, **kw):
                    log.append(("image." + name, a))
                    return self
                return step

        class App:
            def __init__(self, name):
                log.append(("app", name))

            def function(self, **kw):
                log.append(("function", kw))

                def deco(fn):
                    return types.SimpleNamespace(remote=lambda *a: log.append(("remote", a)) or rc_)
                return deco

            def run(self):
                return Ctx()

        class Ctx:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        self.Volume, self.Image, self.App = Volume, Image, App
        self.volume = types.SimpleNamespace(FileEntryType=FileEntryType)
        self.enable_output = Ctx


def test_modal_backend_with_fake_sdk(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "has_credentials", lambda: True)
    m = FakeModal()
    res = M.run(gm_plan(tmp_path), {"gpu": "L4", "volume": "v"}, "exp_t1", modal=m)
    assert res["rc"] == 0 and (res["ckpt"] / "model.safetensors").is_file() and res["remote"]["cleanup"] == "removed"
    fn = next(e[1] for e in m.log if e[0] == "function")
    assert fn["gpu"] == "L4" and fn["secrets"] == [] and fn["volumes"].keys() == {"/vol"}
    puts = [e[2] for e in m.log if e[0] == "put"]
    assert puts == ["jobs/exp_t1/dataset", "jobs/exp_t1/init"]
    argv = next(e[1] for e in m.log if e[0] == "remote")[0]
    assert "--dataset.root=/vol/jobs/exp_t1/dataset" in argv and "--policy.pretrained_path=/vol/jobs/exp_t1/init" in argv
    assert "--policy.push_to_hub=false" in argv
    assert ("image.run_commands", (f"uv venv --python 3.12.12 /opt/venv",
                                   "uv pip sync --python /opt/venv/bin/python --require-hashes /opt/requirements.lock")) in m.log
    assert m.log[-1] == ("remove", "jobs/exp_t1", True)
    m = FakeModal(rc=1)
    assert M.run(gm_plan(tmp_path), {}, "j", modal=m)["rc"] == 1 and m.log[-1][0] == "remove"


def test_modal_refuses_without_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "has_credentials", lambda: False)
    with pytest.raises(SystemExit, match="no Modal credentials"):
        M.run(gm_plan(tmp_path), {}, "j", modal=FakeModal())


def test_container_docker_cannot_start_is_removed(monkeypatch, tmp_path):
    """A train job whose container docker creates but cannot start (nvml error) leaves nothing behind."""
    import docker

    import launch as L
    made = []

    class C:
        def __init__(self, kw):
            self.kw, self.removed = kw, False

        def remove(self, force=False):
            self.removed = True

    class Coll:
        def run(self, image, command=None, **kw):
            made.append(C(kw))
            raise docker.errors.APIError("500", explanation="nvml error: gpu requires reset")

        def list(self, all=False, filters=None):
            k, v = filters["label"].split("=")
            return [c for c in made if c.kw["labels"].get(k) == v]

    monkeypatch.setattr(L.docker, "from_env", lambda: types.SimpleNamespace(containers=Coll()))
    monkeypatch.setenv("REVOLVE_GPU_PREFLIGHT", "0")
    monkeypatch.setenv("REVOLVE_GPU_MACHINE_HOST", "user@gpu-host")
    jd = L.REPO / "runs" / "training_tests" / f"unit_{os.getpid()}" / "jobs" / "train_ckpt"   # job dirs: under runs/
    try:
        with pytest.raises(docker.errors.APIError):
            L.start_train(tmp_path / "ds", jd.parent.parent / "ckpt", steps=1, batch_size=1, backend="gpu_machine",
                          job_dir=jd)
    finally:
        shutil.rmtree(jd.parent.parent, ignore_errors=True)
    with pytest.raises(docker.errors.APIError):
        L.in_image(["true"])
    assert len(made) == 2 and all(c.removed for c in made)


def test_gpu_machine_host_from_env(monkeypatch):
    import launch as L
    monkeypatch.delenv(L.GPU_HOST_ENV, raising=False)
    with pytest.raises(RuntimeError, match=L.GPU_HOST_ENV):
        L.gpu_machine_settings({"scratch": "~/revolve_scratch"})
    monkeypatch.setenv(L.GPU_HOST_ENV, "user@gpu-host")
    assert L.gpu_machine_settings({"scratch": "~/s"}) == {"scratch": "~/s", "host": "user@gpu-host"}
    assert L.gpu_machine_settings({"host": "a@b"})["host"] == "a@b"
