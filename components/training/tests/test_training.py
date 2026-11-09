"""Real training and serve jobs in revolve-training (conftest.py: fixtures, needs). One session fixture runs, in
two threads (the laptop GPU and the GPU machine work at the same time):
- local:  t0 = from_base on turn_0 (40 steps, batch 2, /status polled over HTTP while it trains);
          then t1 = finetune_last on turn_1 from t0 and t2 = finetune_recovery_only on turn_1 from t0 (in
          parallel); then serve t0 on an internal network + one inference request from a client container
- gpu_machine (after t0): finetune_last on turn_1 from t0 on the machine (/status polled), checkpoint back,
          the machine's scratch dir and docker images checked afterwards.
"""
import json
import math
import os
import subprocess
import threading
import time

import docker
import httpx
import pytest

from conftest import GPU_MACHINE, REPO, RUN, load

import launch as L

STEPS, BATCH, LOG_FREQ = 40, 2, 5


def poll(job, seen, stop):
    """GET /status every 0.5 s while the job runs; keeps the samples that show a step and a loss."""
    url = None
    while not stop.is_set():
        try:
            url = url or f"http://{job.ip()}:8000/status"
            p = httpx.get(url, timeout=2).json().get("progress", {})
            if "step" in p:
                seen.append({k: p.get(k) for k in ("phase", "step", "total_steps", "loss", "eta_s", "gpu_mem_gb")})
        except Exception:
            pass
        stop.wait(0.5)


def train(name, dataset, **kw):
    seen, stop = [], threading.Event()
    job = L.start_train(dataset, RUN / name / "ckpt", steps=kw.pop("steps", STEPS), batch_size=BATCH,
                        log_freq=LOG_FREQ, timeout=1800, **kw)
    th = threading.Thread(target=poll, args=(job, seen, stop), daemon=True)
    th.start()
    try:
        res = L.finish(job)
    finally:
        stop.set()
        th.join()
    res["polled"] = seen
    res["ckpt"] = RUN / name / "ckpt"
    return res


def ssh(cmd):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", GPU_MACHINE, cmd], capture_output=True, text=True, timeout=60)


def remote_images():
    return sorted(ssh("docker images -q --no-trunc | sort -u").stdout.split())


@pytest.fixture(scope="session")
def runs(datasets, numbers):
    out, errs = {}, []
    t0_done = threading.Event()

    def guard(f):
        def g():
            try:
                f()
            except BaseException as e:      # surfaced by the tests below
                errs.append(e)
            finally:
                t0_done.set()
        return g

    def local():
        out["t0"] = train("t0", datasets["turn_0"], mode="from_base")
        t0_done.set()
        prev = out["t0"]["ckpt"]
        ths = [threading.Thread(target=lambda: out.__setitem__("t1", train("t1", datasets["turn_1"], mode="finetune_last",
                                                                            prev=prev, steps=20))),
               threading.Thread(target=lambda: out.__setitem__("t2", train("t2", datasets["turn_1"],
                                                                            mode="finetune_recovery_only", prev=prev,
                                                                            steps=10)))]
        [t.start() for t in ths]
        [t.join() for t in ths]
        out["serve"] = serve(prev, datasets["turn_0"])

    def remote():
        if not GPU_MACHINE:
            return
        out["gm_images_before"] = remote_images()
        t0_done.wait()
        if "t0" not in out or out["t0"]["exit"] != 0:
            return
        out["gm"] = train("gm", datasets["turn_1"], mode="finetune_last", prev=out["t0"]["ckpt"], steps=STEPS,
                          backend="gpu_machine", settings={"host": GPU_MACHINE, "scratch": "~/revolve_scratch"})
        out["gm_ls"] = ssh("ls -la ~/revolve_scratch 2>&1; echo; docker ps -a --format '{{.Names}}'")
        out["gm_images_after"] = remote_images()

    t = time.time()
    ths = [threading.Thread(target=guard(local)), threading.Thread(target=guard(remote))]
    [x.start() for x in ths]
    [x.join() for x in ths]
    out["wall_s"] = round(time.time() - t, 1)
    numbers["wall_s"] = out["wall_s"]
    if errs:
        raise errs[0]
    return out


def serve(ckpt, dataset):
    net = f"revolve_training_test_{os.getpid()}"
    cli = docker.from_env()
    cli.networks.create(net, internal=True, labels={L.LABEL: "training"})
    t = time.time()
    job = L.start_serve(ckpt, RUN / "serve" / "job", net, name=f"{net}_server")
    out = {}
    try:
        while time.time() - t < 300:
            st = load(job.dir / "status.json") if (job.dir / "status.json").is_file() else {}
            if st.get("progress", {}).get("ready"):
                break
            job.container.reload()
            assert job.container.status == "running", job.container.logs()[-3000:]
            time.sleep(1)
        out["ready_s"] = round(time.time() - t, 1)
        code, log = L.in_image(["python3", "/repo/components/training/src/training/serve_check.py", f"{net}_server:8080",
                                L.cpath(ckpt), dataset], network=net, timeout=600)
        out["client"] = dict(code=code, **(json.loads(log.strip().splitlines()[-1]) if code in (0, 1) else {"log": log[-3000:]}))
        out["status"] = load(job.dir / "status.json")
        out["ports"] = job.container.attrs["HostConfig"].get("PortBindings")
    finally:
        out["stop"] = L.stop(job)
        cli.networks.get(net).remove()
    return out


def losses(res):
    return res["result"]["progress"]["losses"]


def stdout(res):
    return (res["dir"] / "stdout.log").read_text()


def test_image_pins():
    code, out = L.in_image(["bash", "-c", "grep -E '^(lerobot|torch|transformers|modal|fastapi)==' /opt/pip-freeze.txt"])
    assert code == 0 and "lerobot==0.6.1" in out and "torch==2.11.0" in out and "transformers==5.5.4" in out, out


def test_status_shows_step_and_loss_while_training(runs, numbers):
    seen = runs["t0"]["polled"]
    live = [s for s in seen if s["loss"] is not None and s["step"] < STEPS]
    assert live, seen[-5:]
    assert all(s["total_steps"] == STEPS for s in live)
    numbers["status_samples"] = dict(n=len(seen), first_with_loss=live[0], last=seen[-1])


def test_loss_finite_and_falling(runs, numbers):
    r = runs["t0"]
    assert r["exit"] == 0, stdout(r)[-3000:]
    ls = [l for _, l in losses(r)]
    assert len(ls) == STEPS // LOG_FREQ and all(math.isfinite(x) for x in ls), ls
    assert sum(ls[-3:]) / 3 < ls[0], ls
    numbers["t0"] = dict(losses=losses(r), wall_s=r["wall_s"], s_per_step=r["result"]["progress"]["s_per_step"],
                         gpu_mem_gb=r["result"]["progress"].get("gpu_mem_gb"))


def test_checkpoint_written(runs):
    ck = runs["t0"]["ckpt"]
    for f in ("config.json", "model.safetensors", "train_config.json", "policy_preprocessor.json",
              "policy_postprocessor.json", "train.json"):
        assert (ck / f).is_file(), f
    info = load(ck / "train.json")
    assert info["mode"] == "from_base" and info["init"] == "lerobot/smolvla_base" and info["step"] == f"{STEPS:06d}"
    import train as T
    assert T.sha256(ck / "model.safetensors") == info["model_sha256"]
    assert not (ck.parent / "ckpt.partial").exists()


def test_finetune_last_starts_from_previous_weights(runs, numbers):
    r, base = runs["t1"], runs["t0"]
    assert r["exit"] == 0, stdout(r)[-3000:]
    info = load(r["ckpt"] / "train.json")
    assert info["mode"] == "finetune_last" and info["init"] == L.cpath(runs["t0"]["ckpt"]) and info["episodes"] is None
    assert "dataset.num_episodes=2" in stdout(r)
    # same first-window loss from base was losses(base)[0]; carried-over weights start well below it
    assert losses(r)[0][1] < losses(base)[0][1], (losses(r), losses(base))
    numbers["t1"] = dict(losses=losses(r), wall_s=r["wall_s"])


def test_recovery_only_selects_recovery_episodes(runs, datasets, numbers):
    r = runs["t2"]
    assert r["exit"] == 0, stdout(r)[-3000:]
    info = load(r["ckpt"] / "train.json")
    assert info["mode"] == "finetune_recovery_only" and info["episodes"] == [1]
    code, out = L.in_image(["python3", "-c", "import json; from lerobot.datasets import LeRobotDatasetMetadata as M; "
                            f"m = M('revolve/dataset', root='{L.cpath(datasets['turn_1'])}'); "
                            "print(json.dumps([[int(e['episode_index']), e['labels/source'], int(e['length'])] "
                            "for e in m.episodes]))"])
    assert code == 0, out
    eps = json.loads(out.strip().splitlines()[-1])
    assert [e[0] for e in eps if e[1] == "recovery"] == [1] and [e[1] for e in eps] == ["demo", "recovery"]
    assert f"dataset.num_frames={eps[1][2]} " in stdout(r) and "dataset.num_episodes=1" in stdout(r)
    numbers["t2"] = dict(episodes=info["episodes"], frames=eps[1][2], wall_s=r["wall_s"])


def test_serve_answers_one_inference(runs, numbers):
    s = runs["serve"]
    assert s["client"]["code"] == 0 and s["client"]["ok"], s["client"]
    assert s["client"]["n_actions"] == 30 and s["client"]["action_dim"] == 11
    assert s["status"]["progress"]["ready"] and s["status"]["progress"]["chunks"] >= 1
    assert not s["ports"]                           # nothing published on the host
    assert s["stop"]["result"]["state"] == "cancelled"
    numbers["serve"] = dict(ready_s=s["ready_s"], latency_s=s["client"]["latency_s"], first=s["client"]["first"])


def test_serve_refuses_a_non_internal_network(tmp_path):
    with pytest.raises(ValueError, match="not internal"):
        L.start_serve(tmp_path, tmp_path / "job", "bridge")


@pytest.mark.skipif(not GPU_MACHINE, reason="REVOLVE_GPU_MACHINE empty")
def test_gpu_machine_round_trip(runs, numbers):
    r = runs["gm"]
    assert r["exit"] == 0, stdout(r)[-3000:]
    info = load(r["ckpt"] / "train.json")
    rem = info["remote"]
    assert info["backend"] == "gpu_machine" and info["init"] == L.cpath(runs["t0"]["ckpt"]) and rem["cleanup"] == ["CLEAN"]
    import train as T
    assert T.sha256(r["ckpt"] / "model.safetensors") == info["model_sha256"]
    ls = [l for _, l in losses(r)]
    assert len(ls) == STEPS // LOG_FREQ and all(math.isfinite(x) for x in ls)
    assert any(s["loss"] is not None and s["step"] < STEPS for s in r["polled"]), r["polled"][-5:]
    job_dir = rem["dir"].rsplit("/", 1)[-1]
    assert job_dir not in runs["gm_ls"].stdout, runs["gm_ls"].stdout
    assert runs["gm_images_after"] == runs["gm_images_before"]
    numbers["gpu_machine"] = dict(remote=rem, losses=losses(r), wall_s=r["wall_s"], ls=runs["gm_ls"].stdout.strip(),
                                  s_per_step=r["result"]["progress"]["s_per_step"],
                                  gpu_mem_gb=r["result"]["progress"].get("gpu_mem_gb"))
