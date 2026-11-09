"""The manager: runs the experiment loop (plan.py) one step at a time, each step in its own container.

  python -m manager.loop <experiment.yaml>          (in the manager container; ./revolve run / resume start it)

Containers: the manager image (deps only; the repo is mounted at its HOST path, rw, so the bind mounts the components'
launchers hand to the docker socket are host paths), as the host uid with the docker socket's group, /etc/passwd +
/etc/group read-only (the training gpu_machine backend looks the uid's home up), on docker's default bridge: the
manager polls each step's wrapper at http://<step container IP>:8000/status (no published ports, as every component
job). Step container revolve-<exp>-t<k>-<step>-a<n>, labels revolve.exp / revolve.role=step / revolve.step; what it
starts carries revolve.parent=<its name> (steps.py).

Monitoring: a step whose /status (progress + last line) and newest sub-job status.json under its output dir do not
change for stall_timeout_s is cancelled (POST /cancel, then stopped) and its children removed: attempt "stalled".
Failure policy: a failed or stalled step is retried once (the components resume from their manifests), then the
experiment stops with status failed and the error in summary.json. Resume: state.json is written before a step's
container starts; a restarted manager re-attaches to a running step container, or reads the result.json of one that
finished meanwhile; a step that is done is never run again.
GPU preflight: before a step that needs the GPU (GPU_STEPS; train only on the local backend) and every
REVOLVE_GPU_CHECK_S (60) while it runs, nvidia-smi in a throwaway manager-image container (revolve_common.gpu). A GPU
in a broken state (ERR!, "requires reset", docker cannot start a GPU container) stops the experiment at once with
stop_reason gpu_unhealthy; the step is not counted as failed, so `./revolve resume` after a reboot carries on.
"""
from __future__ import annotations

import json
import os
import pwd
import sys
import time
from pathlib import Path

import docker
import httpx
from docker.types import DeviceRequest

from revolve_common import containers, gpu
from revolve_common import job as J
from revolve_common import specs

from . import plan as P

REPO = Path(__file__).resolve().parents[4]
IMAGE = os.environ.get("REVOLVE_MANAGER_IMAGE", "revolve-manager:latest")
SOCK = "/var/run/docker.sock"
GPU_STEPS = ("generate", "starts", "recover", "eval_heldout", "eval_probe")
WATCH = ("status.json", "*/status.json", "jobs/*/status.json", "*/jobs/*/status.json", "*/*/jobs/*/status.json",
         "*/*/*/jobs/*/status.json")


def say(msg):
    print(f"{time.strftime('%H:%M:%S')} manager: {msg}", flush=True)


def container_kw(env=None):
    """The manager's and the steps' container settings (see the module doc)."""
    home = pwd.getpwuid(os.getuid()).pw_dir
    return dict(detach=True, user=f"{os.getuid()}:{os.getgid()}", group_add=[str(os.stat(SOCK).st_gid)],
                working_dir=str(REPO),
                environment={"PYTHONPATH": f"{REPO}/common:{REPO}/components/manager/src", "PYTHONDONTWRITEBYTECODE": "1",
                             "PYTHONUNBUFFERED": "1", "HOME": home, "XDG_CACHE_HOME": "/tmp/.cache",
                             **{k: v for k, v in os.environ.items() if k.startswith("REVOLVE_")}, **(env or {})},
                volumes={str(REPO): {"bind": str(REPO), "mode": "rw"}, SOCK: {"bind": SOCK, "mode": "rw"},
                         "/etc/passwd": {"bind": "/etc/passwd", "mode": "ro"},
                         "/etc/group": {"bind": "/etc/group", "mode": "ro"},
                         "/etc/localtime": {"bind": "/etc/localtime", "mode": "ro"}})


def step_name(exp, k, step, n):
    return f"revolve-{exp}-t{k}-{step}-a{n}"


def get(dk, name):
    try:
        return dk.containers.get(name)
    except docker.errors.NotFound:
        return None


def ip(c):
    return next((n["IPAddress"] for n in c.attrs["NetworkSettings"]["Networks"].values() if n.get("IPAddress")), None)


def remove_children(dk, parent):
    for c in dk.containers.list(all=True, filters={"label": f"revolve.parent={parent}"}):
        try:
            c.stop(timeout=10)
            c.remove(force=True)
        except docker.errors.NotFound:
            pass
    for n in dk.networks.list(filters={"label": f"revolve.parent={parent}"}):
        try:
            n.remove()
        except docker.errors.APIError:
            pass


def stop_step(dk, name):
    """Cancel through the wrapper, then stop; children removed. Also used by ./revolve stop."""
    c = get(dk, name)
    if c is not None:
        try:
            if c.status == "running" and ip(c):
                httpx.post(f"http://{ip(c)}:8000/cancel", timeout=5)
        except httpx.HTTPError:
            pass
        try:
            c.stop(timeout=15)
            c.remove(force=True)
        except docker.errors.NotFound:
            pass
    remove_children(dk, name)


def newest(watch):
    d = Path(watch)
    return max((p.stat().st_mtime for g in WATCH for p in d.glob(g)), default=0.0)


class GpuUnhealthy(Exception):
    pass


class Manager:
    def __init__(self, spec_path):
        self.spec = Path(spec_path).resolve()
        self.r = specs.load_experiment(self.spec)
        self.e = self.r.experiment
        self.paths = P.Paths(REPO / "runs" / self.e.name)
        self.dk = docker.from_env()
        self.poll_s = float(os.environ.get("REVOLVE_POLL_S", "5"))
        self.gpu_check_s = float(os.environ.get("REVOLVE_GPU_CHECK_S", "60"))
        f = self.paths.exp / "state.json"
        self.state = P.read(f) or P.new_state(self.e.name, self.spec)
        if self.state["spec_sha256"] != P.sha256(self.spec):
            raise SystemExit(f"{self.spec} changed since this experiment started (state.json spec_sha256); "
                             f"restore it or start a new experiment under another name")

    def save(self):
        P.save(self.state, self.paths)

    def run(self):
        st = self.state
        st["status"], st["error"] = "running", None
        self.save()
        say(f"experiment {self.e.name}: task {self.r.task.name}, max_turns {self.e.max_turns}, "
            f"training {self.e.training.mode} on {self.e.training.backend}")
        try:
            while True:
                act = P.next_action(st, self.e.max_turns)
                if act[0] == "stop":
                    st["status"] = "failed" if act[1] == "failed" else "done"
                    st["stop_reason"] = act[1]
                    say(f"stop: {act[1]}" + (f" ({st['error']})" if st["error"] else ""))
                    break
                if act[0] == "turn":
                    st["turns"].append(P.new_turn(act[1]))
                    say(f"turn {act[1]}")
                elif act[0] == "skip":
                    s = st["turns"][act[1]]["steps"][act[2]]
                    s.update(state="skipped", reason=act[3])
                    say(f"turn {act[1]} {act[2]}: skipped ({act[3]})")
                else:
                    self.run_step(act[1], act[2])
                self.save()
        except P.HeldoutLeak as e:
            st.update(status="failed", stop_reason="heldout_leak", error=f"held-out data would reach training: {e}")
            say(st["error"])
        except GpuUnhealthy as e:
            st.update(status="failed", stop_reason="gpu_unhealthy", error=str(e))
            for s in st["turns"][-1]["steps"].values():
                if s["state"] == "running":
                    s["state"] = "pending"             # not a failure: resume relaunches it
            say(f"stop: gpu_unhealthy ({e}); ./revolve resume {self.e.name} once the GPU is healthy")
        self.save()
        return 0 if st["status"] == "done" else 1

    def needs_gpu(self, step):
        if os.environ.get("REVOLVE_STUB"):
            return False
        return step in GPU_STEPS or (step == "train" and self.e.training.backend == "local")

    def gpu_preflight(self, k, step):
        try:
            gpu.check_docker(self.dk, IMAGE)
        except gpu.GpuError as e:
            raise GpuUnhealthy(f"turn {k} {step}: {e}") from None

    # -------------------------------------------------- one step (launch or re-attach, retry once)
    def run_step(self, k, step):
        s = self.state["turns"][k]["steps"][step]
        while True:
            att = s["attempts"][-1] if s["attempts"] else None
            if att is None or att["state"] != "running":
                failed = [a for a in s["attempts"] if a["state"] in ("failed", "stalled", "cancelled")]
                if len(failed) >= 2:
                    s["state"] = "failed"
                    self.state["error"] = f"turn {k} {step}: failed twice (last: {failed[-1].get('error')})"
                    return
                params = P.step_params(self.state, k, step, self.r, self.paths)
                if params is None:
                    s.update(state="skipped", reason="nothing to do (no new episodes / no probe failure)")
                    say(f"turn {k} {step}: skipped ({s['reason']})")
                    return
                if self.needs_gpu(step):
                    self.gpu_preflight(k, step)
                att = self.launch(k, step, len(s["attempts"]), params, s)
            else:
                say(f"turn {k} {step}: re-attaching to {att['container']}")
            if att["state"] == "running":
                self.watch(att, step)
            if att["state"] == "succeeded":
                outputs, err = P.collect(step, att["params"], self.paths)
                if err is None:
                    s.update(state="done", outputs=outputs)
                    say(f"turn {k} {step}: done {json.dumps({x: y for x, y in outputs.items() if x not in ('episodes', 'table', 'failures')})}")
                    return
                att.update(state="failed", error=err)
            say(f"turn {k} {step} attempt {att['n']}: {att['state']} ({att.get('error')})")
            self.save()

    def launch(self, k, step, n, params, s):
        name = step_name(self.e.name, k, step, n)
        jd = self.paths.step_dir(k, step, n)
        jd.mkdir(parents=True, exist_ok=True)
        for f in (J.STATUS, J.RESULT, J.LOG):
            (jd / f).unlink(missing_ok=True)
        old = get(self.dk, name)
        if old is not None:
            old.remove(force=True)
        J.write_model(jd / J.JOB, J.Job(id=name, component="manager", command=["python", "-m", "manager.steps", str(jd)],
                                        params={**params, "exp": self.e.name, "step": step, "turn": k, "attempt": n}))
        att = dict(n=n, container=name, job_dir=str(jd), state="running", started_at=time.time(), params=params)
        s["attempts"].append(att)
        s["state"] = "running"
        self.save()                                   # before the container exists: a resume finds it or relaunches
        kw = container_kw()
        if step.startswith("eval_") and not os.environ.get("REVOLVE_STUB"):
            # nvidia-smi only (evaluate sizes its parallelism from the free GPU memory)
            kw["device_requests"] = [DeviceRequest(count=-1, capabilities=[["gpu"]])]
            kw["environment"]["NVIDIA_DRIVER_CAPABILITIES"] = "utility"
        try:
            containers.run(self.dk, IMAGE, ["python", "-m", "revolve_common.wrapper", "--job-dir", str(jd), "--linger-s", "2"],
                           name=name, labels={"revolve.exp": self.e.name, "revolve.role": "step",
                                              "revolve.step": f"t{k}/{step}", "revolve.attempt": str(n)}, **kw)
        except docker.errors.APIError as e:            # created but not started: containers.run removed it
            att.update(state="failed", error=f"docker could not start {name}: {e.explanation or e}",
                       finished_at=time.time(), wall_s=0.0)
            return att
        say(f"turn {k} {step} attempt {n}: started {name}")
        return att

    def poll(self, c, jd):
        try:
            return httpx.get(f"http://{ip(c)}:8000/status", timeout=5).json()
        except (httpx.HTTPError, ValueError, StopIteration, TypeError):
            st = J.read_status(jd)                     # not up yet / gone: the file the wrapper keeps
            return st.model_dump(mode="json") if st else {}

    def watch(self, att, step=None):
        name, jd = att["container"], Path(att["job_dir"])
        stall = self.e.stall_timeout_s
        last, t_change = None, time.time()
        gpu_step, t_gpu = step is not None and self.needs_gpu(step), time.time()
        while True:
            c = get(self.dk, name)
            if c is None or c.status in ("exited", "dead"):
                res = J.read_result(jd)
                if c is not None:
                    (jd / "container.log").write_bytes(c.logs())
                    c.remove(force=True)
                remove_children(self.dk, name)
                if res is None:
                    att.update(state="lost" if c is None else "failed",
                               error="container gone, no result.json" if c is None else "exited without result.json")
                else:
                    att.update(state=res.state, returncode=res.returncode, progress=res.progress,
                               error=res.error or (f"exit {res.returncode}" if res.returncode else None))
                att.update(finished_at=time.time(), wall_s=round(time.time() - att["started_at"], 1))
                return
            if gpu_step and time.time() - t_gpu > self.gpu_check_s:
                t_gpu = time.time()
                try:
                    gpu.check_docker(self.dk, IMAGE)
                except gpu.GpuError as e:
                    say(f"{name}: {e}: cancelling")
                    stop_step(self.dk, name)
                    att.update(state="gpu_unhealthy", error=str(e), finished_at=time.time(),
                               wall_s=round(time.time() - att["started_at"], 1))
                    raise GpuUnhealthy(f"{name}: {e}") from None
            st = self.poll(c, jd)
            fp = json.dumps([st.get("progress"), st.get("last_line"), newest(att["params"].get("watch", jd))])
            if fp != last:
                last, t_change = fp, time.time()
                att["progress"] = st.get("progress") or {}
                self.save()
            elif time.time() - t_change > stall:
                say(f"{name}: no progress for {stall:.0f} s: cancelling")
                stop_step(self.dk, name)
                att.update(state="stalled", error=f"stalled: no progress for {stall:.0f} s", finished_at=time.time(),
                           wall_s=round(time.time() - att["started_at"], 1))
                return
            time.sleep(self.poll_s)


def main(argv=None):
    return Manager((argv or sys.argv[1:])[0]).run()


if __name__ == "__main__":
    sys.exit(main())
