"""Host side of ./revolve run | resume | status | stop <experiment> (a name under specs/experiments/ or a yaml path).

  run      builds revolve-manager if missing, starts the manager container (detached) on a NEW experiment;
           its output goes to runs/<exp>/manager.log
  resume   the same on an existing runs/<exp>/state.json (done steps are kept, running ones re-attached);
           accept_spec_change: the yaml was edited since the run started (the manager refuses that otherwise):
           its new sha256 goes into state.json, the old one into spec_changes; done steps are not redone
  status   seed x phase table per turn (first try) + the running step's live progress (training step / loss)
  stop     kills the manager, cancels the running step and every container it started; state "stopped"
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import docker

from revolve_common import specs

from . import loop as L
from . import plan as P

REPO = L.REPO


def spec_of(arg):
    p = Path(arg)
    if p.suffix in (".yaml", ".yml") and p.is_file():
        return p.resolve()
    st = P.read(REPO / "runs" / arg / "state.json")
    if st is not None:                   # an experiment that ran: its spec (tests keep theirs outside specs/)
        return Path(st["spec"])
    p = REPO / "specs" / "experiments" / f"{arg}.yaml"
    if not p.is_file():
        raise SystemExit(f"no experiment {arg!r} ({p})")
    return p


def manager_name(exp):
    return f"revolve-{exp}-manager"


def start(arg, resume, accept_spec_change=False):
    spec = spec_of(arg)
    r = specs.load_experiment(spec)
    exp = r.experiment.name
    os.environ[specs.ROBOT_ENV] = r.robot.name     # the experiment's robot, passed to the manager (container_kw)
    L.set_variant_env(r.experiment)                # and its datagen variant
    paths = P.Paths(REPO / "runs" / exp)
    have = (paths.exp / "state.json").is_file()
    if resume and not have:
        raise SystemExit(f"nothing to resume: no {paths.exp / 'state.json'} (use ./revolve run {arg})")
    if not resume and have:
        raise SystemExit(f"{paths.exp} exists: ./revolve resume {exp} (or move it away for a fresh run)")
    if resume and accept_spec_change:
        st = P.read(paths.exp / "state.json")
        new = P.sha256(spec)
        if st["spec_sha256"] != new:
            st.setdefault("spec_changes", []).append(dict(old=st["spec_sha256"], new=new,
                                                          at=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
            st["spec_sha256"] = new
            P.save(st, paths)
    dk = docker.from_env()
    m = L.get(dk, manager_name(exp))
    if m is not None and m.status == "running":
        raise SystemExit(f"{manager_name(exp)} is already running")
    if m is not None:
        m.remove(force=True)
    try:
        dk.images.get(L.IMAGE)
    except docker.errors.ImageNotFound:
        if subprocess.call(["bash", str(REPO / "components" / "manager" / "build.sh")]):
            raise SystemExit("revolve build manager failed")
    paths.exp.mkdir(parents=True, exist_ok=True)
    log = paths.exp / "manager.log"
    with open(log, "a") as f:
        f.write(f"\n==== {'resume' if resume else 'run'} {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
    dk.containers.run(L.IMAGE, ["sh", "-c", f'exec python -m manager.loop "$0" >> "$1" 2>&1', str(spec), str(log)],
                      name=manager_name(exp), labels={"revolve.exp": exp, "revolve.role": "manager"}, auto_remove=True,
                      **L.container_kw())
    print(f"{'resumed' if resume else 'started'} {exp}: container {manager_name(exp)}, log {log.relative_to(REPO)}; "
          f"./revolve status {exp}")
    return 0


def exp_of(arg):
    p = Path(arg)
    return specs.load_experiment(p).experiment.name if p.suffix in (".yaml", ".yml") and p.is_file() else arg


def stop(arg):
    exp = exp_of(arg)
    dk = docker.from_env()
    m = L.get(dk, manager_name(exp))
    if m is not None:
        m.remove(force=True)
    for c in dk.containers.list(all=True, filters={"label": [f"revolve.exp={exp}", "revolve.role=step"]}):
        L.stop_step(dk, c.name)
    for c in dk.containers.list(all=True, filters={"label": f"revolve.exp={exp}"}):    # orphans of a step
        c.remove(force=True)
    paths = P.Paths(REPO / "runs" / exp)
    st = P.read(paths.exp / "state.json")
    if st is not None:
        for t in st["turns"]:
            for s in t["steps"].values():
                for a in s["attempts"]:
                    if a["state"] == "running":
                        a.update(state="stopped", error="./revolve stop")   # not a failure: resume relaunches it
                if s["state"] == "running":
                    s["state"] = "pending"
        st["status"] = "stopped"
        P.save(st, paths)
    print(f"stopped {exp}")
    return 0


def fmt_table(name, o, phases):
    lines = [f"    {name}: all_pass {o['all_pass']} ({o['first_try_passed']}/{o['rollouts']} first try)",
             "      seed     " + " ".join(f"{p:>9}" for p in phases)]
    for seed, rows in o["table"].items():
        for r, ph in rows.items():
            lines.append(f"      {seed + '_' + r:<8} " + " ".join(f"{(ph or {}).get(p, '-'):>9}" for p in phases)
                         + ("" if ph else "  (no verdict)"))
    return lines


def status(arg):
    exp = exp_of(arg)
    paths = P.Paths(REPO / "runs" / exp)
    st = P.read(paths.exp / "state.json")
    if st is None:
        raise SystemExit(f"no experiment state at {paths.exp}")
    phases = [p.name for p in specs.load_experiment(st["spec"]).task.phases]
    try:
        m = L.get(docker.from_env(), manager_name(exp))
        mstate = m.status if m else "not running"
    except docker.errors.DockerException:
        mstate = "unknown"
    out = [f"{exp}: {st['status']}" + (f", stop: {st['stop_reason']}" if st["stop_reason"] else "") +
           f" (manager {mstate})" + (f"\n  error: {st['error']}" if st["error"] else "")]
    for t in st["turns"]:
        s = t["steps"]
        out.append(f"  turn {t['turn']}: " + ", ".join(
            f"{n} {x['state']}" + (f" x{len(x['attempts'])}" if len(x["attempts"]) > 1 else "") for n, x in s.items()))
        d, tr = s["dataset"]["outputs"], s["train"]["outputs"]
        if d or tr:
            out.append(f"    data: {d.get('episodes', '-')} episodes ({d.get('new_episodes', 0)} new); "
                       f"train: {tr.get('steps', '-')} steps, final loss {tr.get('final_loss', '-')}")
        for n in ("eval_heldout", "eval_probe"):
            if s[n]["outputs"]:
                out += fmt_table(n.split("_", 1)[1], s[n]["outputs"], phases)
        for n, x in s.items():
            if x["state"] == "running" and x["attempts"]:
                a = x["attempts"][-1]
                live = (P.read(Path(a["job_dir"]) / "status.json") or {})
                prog = {k: v for k, v in (live.get("progress") or {}).items() if k not in ("losses", "running")}
                out.append(f"    live {n} (attempt {a['n']}, {a['container']}, {time.time() - a['started_at']:.0f} s): {prog}")
                if live.get("last_line"):
                    out.append(f"      last line: {live['last_line'][-160:]}")
    print("\n".join(out))
    return 0
