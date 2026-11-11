"""One closed-loop policy rollout (host side, docker SDK). Ported from the earlier pipeline's evaluate.py episode():
per rollout an --internal docker network with its own ROS_DOMAIN_ID (the manager never joins
ROS), three containers on it, then every container and the network are removed:

  serve   revolve-training `serve` (Rosetta's lerobot_rosetta.policy_server, GPU; components/training/launch.py
          start_serve), clients reach <tag>-serve:8080. Not started for the oracle.
  client  revolve-dataset (ROS 2 Jazzy + Rosetta from ThirdParty/rosetta/install): Rosetta's own
          policy_runner_node (ros2 launch rosetta_ros policy_runner_launch.py, the policy's runner parameters)
          + src/evaluation/rosetta_goal.py (one RunPolicy goal), or for the oracle src/evaluation/dummy_policy.py
          (the start's own expert actions at the contract rate on the same action topic).
  sim     revolve-simulator-ros: components/simulator cli.py rollout (closed_loop_system + the live phase
          validator in policy mode + ros_relay.py), the client as its static DDS peer (an internal network
          drops multicast discovery), FASTDDS_BUILTIN_TRANSPORTS=UDPv4 (each container has its own /dev/shm).

  rec = run(start_dir, out, seed=S, policy="smolvla", ckpt=CKPT)        # -> out/rollout.json
  rec = run(start_dir, out, seed=S, policy="oracle")
  rec = run(start_dir, out2, seed=S, policy=..., restore=(out, "open"))   # a retry from out's open_start.npz

Readiness goes through files on the shared runs/ mount: the serve wrapper's status.json (ready), the sim's
<out>/relay_ready and the client's <out>/client/runner_ready (the world stays paused until both exist).
Time: the sim budget is task.yaml `rollout` (sim time: wait_s before the first action, budget_s after it,
after_success_s run-on); the wall-clock limit is derived from it and the measured real-time factor
(wall_limit), never a fixed number.

out/ (= the recovery input contract): phase_report.json, phase_states/<phase>_start(_next).npz, episode.json,
live.json, closed_loop.json, state_log.npz, actions.npz, summary.json, relay.json, gz_sim.log, driver.log;
client/ (runner_params.yaml, runner_node.log, goal.log, runner.json or dummy_policy.json, client.log);
jobs/sim/, jobs/serve/ (job.json, status.json, result.json, stdout.log, container.log); rollout.json.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import os
import shlex
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path

import docker
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "components" / "simulator" / "src" / "sim"))
import spec as sim_spec  # noqa: E402  (the simulator's specs bridge)

from revolve_common import job as J  # noqa: E402


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = sys.modules.setdefault(name, importlib.util.module_from_spec(s))
    if not hasattr(m, "REPO"):
        s.loader.exec_module(m)
    return m


sim_launch = _load("sim_launch", REPO / "components" / "simulator" / "launch.py")
train_launch = _load("training_launch", REPO / "components" / "training" / "launch.py")
data_launch = _load("dataset_launch", REPO / "components" / "dataset" / "launch.py")
cpath = sim_launch.cpath

SIM_ROS_IMAGE = os.environ.get("REVOLVE_SIM_ROS_IMAGE", "revolve-simulator-ros:latest")
CLIENT_SRC = "/repo/components/evaluation/src/evaluation"
LABEL = "revolve.rollout"
POLICIES = ("smolvla", "oracle")
# Rosetta d552e34's policy_runner defaults, as the earlier pipeline's policy config wrote them out
RUNNER_PARAMS = dict(actions_per_chunk=30, chunk_size_threshold=0.95, aggregate_fn_name="weighted_average")
SERVE_READY_S = 300            # legacy: "RosettaPolicyServer started on" within 300 s
RUNNER_JSON_S = 90             # rosetta_goal.py: cancel the goal + runner.json after the sim's summary.json
SETUP_S = 240                  # container start, models, gz world load, client up (the sim waits up to 300 s)
SAFETY, RTF_FLOOR = 2.0, 0.03  # wall limit = sim budget / measured RTF x SAFETY (+ SETUP_S)
_DOMAINS = itertools.count(int(os.environ.get("REVOLVE_ROS_DOMAIN0", "11")))
_DOMAIN_LOCK = threading.Lock()


def next_domain():
    """A ROS_DOMAIN_ID per rollout (11..110 cycling; each rollout also has its own network)."""
    with _DOMAIN_LOCK:
        return 11 + (next(_DOMAINS) - 11) % 100


def sim_budget_s(task_rollout, budget_s=None, wait_s=None, after_success_s=None):
    """The rollout's sim time at most: wait before the first action + budget after it + the run-on."""
    b = task_rollout.budget_s if budget_s is None else budget_s
    w = task_rollout.wait_s if wait_s is None else wait_s
    a = task_rollout.after_success_s if after_success_s is None else after_success_s
    return w + b + (a or 0.0)


def wall_limit(sim_s, rtf):
    """Wall clock [s] the gz run may take from its unpause: sim_s at the measured real-time factor, x SAFETY."""
    return round(sim_s / max(float(rtf), RTF_FLOOR) * SAFETY, 1)


def read(p):
    p = Path(p)
    try:
        return json.loads(p.read_text()) if p.is_file() else None
    except ValueError:
        return None


def server_latency(log):
    """Per-observation times from the policy server log (legacy evaluate.py server_latency)."""
    inf, tot = [], []
    for ln in log.splitlines():
        if "Preprocessing and inference took" in ln:
            inf.append(float(ln.split("took ")[1].split("s")[0]) * 1000)
        elif "| Total time:" in ln and "Observation" in ln:
            tot.append(float(ln.split("Total time: ")[1].split("ms")[0]))
    q = lambda v, p: round(sorted(v)[min(len(v) - 1, int(p * len(v)))], 1) if v else None
    return dict(inferences=len(inf), inference_ms=dict(median=q(inf, 0.5), p90=q(inf, 0.9), max=q(inf, 1.0)),
                server_total_ms=dict(median=q(tot, 0.5), p90=q(tot, 0.9), max=q(tot, 1.0))) if inf else None


def _client_kw(out, tag, net, domain):
    return dict(detach=True, user=f"{os.getuid()}:{os.getgid()}", working_dir="/tmp", name=f"{tag}-client",
                labels={LABEL: tag, sim_launch.LABEL: "evaluation"}, network=net, shm_size="1g",
                environment={"HOME": "/tmp", "REPO": "/repo", "PYTHONDONTWRITEBYTECODE": "1", "ROS_LOG_DIR": "/tmp/ros_log",
                             "ROS_DOMAIN_ID": str(domain), "FASTDDS_BUILTIN_TRANSPORTS": "UDPv4",
                             **data_launch.OFFLINE_ENV},
                volumes={str(REPO): {"bind": "/repo", "mode": "ro"}, str(REPO / "runs"): {"bind": "/repo/runs", "mode": "rw"},
                         str(data_launch.INSTALL): {"bind": "/opt/rosetta/install", "mode": "ro"}})


def client_script(policy, out, start, prompt, contract, ckpt=None, server=None, max_duration_s=3600):
    """The client container's bash script (cwd /tmp: LeRobot's RobotClient writes logs/ in its cwd)."""
    cl = cpath(out / "client")
    src = "source /opt/rosetta/install/setup.bash && cd /tmp && "
    if policy == "oracle":
        return src + (f"exec python3 -B {CLIENT_SRC}/dummy_policy.py {cpath(start)} {cpath(out)} {cl} runner_ready "
                      f"{cpath(contract)} > {cl}/client.log 2>&1")
    launch = ["ros2", "launch", "rosetta_ros", "policy_runner_launch.py", f"params_file:={cl}/runner_params.yaml",
              f"contract_path:={cpath(contract)}", f"pretrained_name_or_path:={cpath(ckpt)}",
              f"server_address:={server}:8080", "launch_local_server:=false", "use_sim_time:=true",
              "policy_type:=smolvla", "policy_device:=cuda"]
    goal = ["python3", "-B", f"{CLIENT_SRC}/rosetta_goal.py", cl, prompt, str(int(max_duration_s)),
            f"{cpath(out)}/summary.json", cpath(contract)]
    return src + (f"{{ {shlex.join(launch)} > {cl}/runner_node.log 2>&1 & }} && exec {shlex.join(goal)} > {cl}/goal.log 2>&1")


def run(start, out, *, seed, policy="smolvla", ckpt=None, restore=None, budget_s=None, wait_s=None,
        after_success_s=None, rtf=0.1, task=None, say=print, gui=False, video=False) -> dict:
    """One rollout from `start` (a replay-accepted attempt dir, starts.py) into `out` (emptied first). restore:
    (a finished rollout dir, phase) = a retry from its phase_states/<phase>_start.npz. gui: the sim container also
    shows the gz GUI on the host's X display (simulator launch.py gui_kw; not part of the verdict). video: the
    sim also renders the scene camera into <out>/video.mp4 (cli.py rollout --video; not part of the verdict).
    -> rollout.json's record."""
    if policy not in POLICIES:
        raise ValueError(f"policy {policy!r}: one of {POLICIES}")
    if policy == "smolvla" and not (Path(ckpt) / "config.json").is_file():
        raise ValueError(f"no checkpoint at {ckpt}")
    start, out = Path(start).resolve(), Path(out).resolve()
    for p in (start, out, *([Path(ckpt).resolve()] if ckpt else [])):
        if not p.is_relative_to(REPO / "runs"):          # the containers see the repo (ro) and runs/ (rw) only
            raise ValueError(f"{p}: not under {REPO / 'runs'}")
    r = sim_spec.task(task or json.loads((start / "episode.json").read_text())["task"])
    contract = sim_spec.SPECS / "robots" / r.robot.name / r.robot.contract
    sim_s = sim_budget_s(r.task.rollout, budget_s, wait_s, after_success_s)
    wall = wall_limit(sim_s, rtf)
    shutil.rmtree(out, ignore_errors=True)
    (out / "client").mkdir(parents=True)
    tag = f"rollout-{uuid.uuid4().hex[:8]}"
    domain = next_domain()
    dk = docker.from_env()
    t0 = time.time()
    rec = dict(seed=seed, dir=str(out.relative_to(REPO)) if out.is_relative_to(REPO) else str(out), policy=policy,
               ckpt=str(ckpt) if ckpt else None, start=str(start.relative_to(REPO)) if start.is_relative_to(REPO) else str(start),
               restore=None if restore is None else dict(source=str(restore[0]), phase=restore[1]), tag=tag,
               ros_domain_id=domain, sim_budget_s=sim_s, rtf_assumed=rtf, wall_limit_s=wall, timing={}, error=None)
    serve = sim = None
    net = dk.networks.create(f"{tag}-net", internal=True, labels={LABEL: tag})
    try:
        if policy == "smolvla":
            serve = train_launch.start_serve(Path(ckpt).resolve(), out / "jobs" / "serve", net.name, name=f"{tag}-serve")
            while not ((J.read_status(serve.dir) or J.Status(job_id="")).progress or {}).get("ready"):
                serve.container.reload()
                if serve.container.status == "exited":
                    raise RuntimeError("policy server exited before it was ready")
                if time.time() - t0 > SERVE_READY_S:
                    raise RuntimeError(f"policy server not ready within {SERVE_READY_S} s")
                time.sleep(0.5)
            rec["timing"]["serve_ready_s"] = round(time.time() - t0, 1)
            (out / "client" / "runner_params.yaml").write_text(
                yaml.safe_dump(dict(policy_runner=dict(ros__parameters=RUNNER_PARAMS))))
        script = client_script(policy, out, start, r.task.prompt, contract, Path(ckpt).resolve() if ckpt else None,
                               f"{tag}-serve", wall + SETUP_S)
        client = dk.containers.run(data_launch.IMAGE, ["bash", "-c", script], **_client_kw(out, tag, net.name, domain))
        client.reload()
        peer = client.attrs["NetworkSettings"]["Networks"][net.name]["IPAddress"]
        args = ["rollout", "--start", start, "--out", out, "--seed", seed, "--wall-s", wall]
        args += ["--restore", f"{cpath(restore[0])}:{restore[1]}"] if restore else []
        for k, v in (("--budget-s", budget_s), ("--wait-s", wait_s), ("--after-success-s", after_success_s)):
            args += [k, v] if v is not None else []
        args += ["--gui"] if gui else []
        args += ["--video"] if video else []
        sim = sim_launch.start(args, out / "jobs" / "sim", gpu=True, timeout=wall + SETUP_S, network=net.name,
                               image=SIM_ROS_IMAGE, name=f"{tag}-sim", labels={LABEL: tag},
                               env={"ROS_DOMAIN_ID": str(domain), "FASTDDS_BUILTIN_TRANSPORTS": "UDPv4",
                                    "ROS_STATIC_PEERS": peer}, gui=gui)
        say(f"{tag}: seed {seed} {policy}" + (f" restored at {restore[1]}" if restore else "") +
            f", domain {domain}, sim budget {sim_s:.0f} s, wall limit {wall:.0f} s (RTF {rtf:.3f})")
        res = sim_launch.finish(sim)
        sim = None
        rec["timing"]["sim_job_s"] = res["wall_s"]
        rec["sim_exit"] = res["exit"]
        if res["exit"] is None:
            rec["error"] = f"sim job over its wall limit {wall + SETUP_S:.0f} s"
        if policy == "smolvla":
            t1 = time.time()
            while not (out / "client" / "runner.json").exists() and time.time() - t1 < RUNNER_JSON_S:
                client.reload()
                if client.status == "exited":
                    break
                time.sleep(0.5)
    except Exception as e:                       # the stack failed (not the policy): recorded, torn down
        rec["error"] = f"{type(e).__name__}: {e}"
    finally:
        teardown(dk, tag, out, serve, sim, net)
    rec["timing"]["rollout_wall_s"] = round(time.time() - t0, 1)
    rec.update(verdict(out, rec))
    J.write_json(out / "rollout.json", rec)
    return rec


def teardown(dk, tag, out, serve, sim, net):
    """Every container of the rollout (its label and names), then its network; logs kept in out/."""
    if sim is not None:
        sim.container.stop(timeout=20)
        sim_launch.finish(sim)
    if serve is not None:
        train_launch.stop(serve)
    for c in dk.containers.list(all=True, filters={"label": f"{LABEL}={tag}"}) + \
            dk.containers.list(all=True, filters={"name": f"^{tag}-"}):
        try:
            if c.name.endswith("-client"):
                c.stop(timeout=15)           # SIGTERM: the oracle / goal client write their json on the way out
                (out / "client" / "container.log").write_bytes(c.logs())
            c.remove(force=True)
        except docker.errors.NotFound:
            pass
    try:
        net.remove()
    except docker.errors.NotFound:
        pass


def verdict(out, rec):
    """The rollout's verdict from the sim's phase_report.json (+ summary.json timing, runner.json, server log)."""
    pr, s = read(out / "phase_report.json"), read(out / "summary.json") or {}
    ex = s.get("extra", {})
    ro, rp = ex.get("rollout", {}), ex.get("replay", {})
    run_ = read(out / "client" / "runner.json")
    slog = out / "jobs" / "serve" / "stdout.log"
    v = dict(phases=None, failed_phase=None, regenerate=None, failed_checks=[], passed=False,
             end_reason=ro.get("end_reason"), sim_s=ro.get("sim_s"), rtf=rp.get("rtf"), loop_wall_s=rp.get("loop_wall_s"),
             first_action_t=ro.get("first_action_t"), actions_accepted=ro.get("actions_accepted"),
             joint_max=ro.get("joint_max"), hold_check=(ex.get("closed_loop") or {}).get("success"),
             runner=None if run_ is None else {k: run_.get(k) for k in ("accepted", "termination_reason", "status", "error")},
             server=server_latency(slog.read_text(errors="replace")) if slog.is_file() else None)
    if pr is None:
        v["error"] = rec["error"] or f"no phase_report.json (sim exit {rec.get('sim_exit')})"
        return v
    v["phases"] = {p: r["status"] for p, r in pr["phases"].items()}
    res = pr["result"]
    v.update(failed_phase=res["failed_phase"], regenerate=res["regenerate"], failed_checks=res["failed_checks"],
             passed=res["failed_phase"] is None, restore_check=(pr.get("restore_check") or {}).get("pass_"))
    return v
