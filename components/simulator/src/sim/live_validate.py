"""The live phase validator around one gz run (cli.py calls it in revolve-simulator; phases.py has the
phase rules, phase_replay_system.py the in-sim part). Ported from
the earlier pipeline's live_validate.py; the gates and thresholds come from the task spec.

Preparers (write the feed <out>/actions.npz, <out>/episode.json read by ReplaySystem, <out>/live.json):
  prepare_plan(plan run dir, attempt, out, task, feed)   one AutoMoMa plan attempt (replay.py actions):
      feed 1khz = its 1 ms actions; 30hz = what a 30 Hz policy would command (held_feed)
  prepare_scripted(name, out, task)    a scripted episode (kitchen_episodes.py: fridge, drawer)
  prepare_restore(source run dir, phase, out)   the source run from <phase>_start.npz on
      (phase_states/ of a prepare_plan / prepare_scripted run), restored in step 0
finalize(out)
      phase_live.json (the system) + the gz/MuJoCo log check (summary.json) + the offline verdict of
      this run (replays: validation.verdict -> verdict.json) -> phase_report.json; restore runs:
      restore_check.json; policy rollouts (mode "rollout", cli.py rollout): the phases alone. Prints one LIVE line. Exit 0 = every phase passed (and, for a replay, the
      offline verdict accepted; for a restore, the restore check passed), 1 otherwise.
"""
import json
import shutil
from pathlib import Path

import numpy as np

from revolve_common import metrics as M

import phases as ph
import spec
import validation

HERE = Path(__file__).resolve().parent
FEED_HZ = {"1khz": None, "30hz": 30}


def held_feed(a, dt, hz):
    """The recorded 1 ms actions as a policy at `hz` would command them (see the module doc)."""
    n = len(a["grip"])
    frame = np.floor(np.arange(n) * dt * hz + 1e-9) / hz             # the frame each step is in
    src = np.minimum(np.floor(frame / dt + 1e-6).astype(int), n - 1)  # its 1 ms step at or before
    return dict(base=a["base"][src], base_vel=np.zeros_like(a["base_vel"]), arm=a["arm"][src],
                arm_vel=np.zeros_like(a["arm_vel"]), grip=a["grip"][src], grip_force=a["grip_force"][src])


def shift_episode(ep, b, dt, n_left):
    """episode.json for a run that starts at step b of the source: events from b on (the event in
    effect at b moved to 0), checks re-indexed, those of dropped events left out."""
    ev = ep["events"]
    steps = [int(round(t / dt)) for t, _, _ in ev]
    keep = [i for i, k in enumerate(steps) if k > b]
    cur = max(i for i, k in enumerate(steps) if k <= b)
    keep = [cur] + keep
    new = {old: i for i, old in enumerate(keep)}
    out = dict(ep)
    out["events"] = [[0.0 if i == cur else round(ev[i][0] - b * dt, 6), ev[i][1], ev[i][2]] for i in keep]
    out["event_subtask"] = [ep["event_subtask"][i] for i in keep]
    out["checks"] = [[new[i], kind, info] for i, kind, info in ep["checks"] if i in new and i != cur]
    out["steps"], out["duration_s"] = n_left, n_left * dt
    out["restored_from_step"] = b
    return out


def live_json(r, out, label, feed, grasp_pose, mode, **extra):
    obj, part, unit = spec.hold(r)[:3]
    live = dict(feed=feed, label=label, mode=mode, task=r.task.name, obj=obj, part=part, unit=unit,
                world=r.scene.world, grasp_pose_world=np.asarray(grasp_pose, float).tolist(), restore=None,
                policy=None, **extra)
    (Path(out) / "live.json").write_text(json.dumps(live, indent=1))
    return live


def prepare_plan(run, attempt, out, task_name=None, feed="1khz"):
    """replay.py actions + live.json. Exit 1 (verdict.json written, no gz) for a spawn in collision."""
    import replay
    run, out = Path(run), Path(out)
    rc = replay.actions(run, attempt, out, task_name)
    ep = json.loads((out / "episode.json").read_text())
    h = np.load(run / "handoff.npz")
    k = int(np.flatnonzero(h["attempt_id"] == int(attempt))[0])
    if FEED_HZ[feed]:
        a = held_feed(dict(np.load(out / "actions.npz")), ep["control_dt"], FEED_HZ[feed])
        np.savez_compressed(out / "actions.npz", **a)
    live_json(spec.task(ep["task"]), out, f"{run.name}/{attempt}", feed, h["tcp_world"][k, 0], "replay",
              plan_run=str(run), attempt_id=int(attempt))
    return rc


def prepare_scripted(name, out, task_name):
    """A scripted kitchen episode (kitchen_episodes.py) as a fed stream: its plan's actions, events, check
    windows and spans; the grasp target of its first grasp is the approach gate's reference."""
    import kitchen_episodes as ke
    import mobile_panda as mp
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    r = spec.task(task_name)
    sc = ke.episode(name)
    seq = sc.P.actions()
    seq.save(out / "actions.npz")
    obj, part, unit = spec.hold(r)[:3]
    man = json.loads((spec.path(r.task.object.gz_model) / "manifest.json").read_text())
    hd = next(x for x in man["handles"] if x["name"] == r.task.object.handle)
    hold_check = r.task.replay.hold_check
    ev = [[round(t, 6), lab, info] for t, lab, info in sc.P.events]
    ep = dict(plan_run=f"scripted_{name}", attempt_id=0, grasp_id=0, scripted=name, control_dt=mp.CONTROL_DT,
              steps=len(seq), duration_s=sc.P.t, budget_s=sc.info["budget"], contact_rate_hz=spec.SIM["contact_rate_hz"],
              task=r.task.name, object_joint=f"{r.task.object.name}::{r.task.object.joint}",
              object_handle=f"{r.task.object.name}::{hd['link']}::{r.task.object.handle}",
              joint_topic=r.task.object.joint_topic, prompt=r.task.prompt,
              hold=dict(check=hold_check, obj=obj, part=part, unit=unit),
              thresholds=dict(spec.legacy_thresholds(r), grasp_half_width_m=spec.grasp_half_width(r, hd["size"])),
              events=ev, event_subtask=[lab for _, lab, _ in ev], subtasks=list(dict.fromkeys(lab for _, lab, _ in ev)),
              checks=[[i, kind, info] for i, kind, info in sc.checks], spans=getattr(sc, "spans", []))
    (out / "episode.json").write_text(json.dumps(ep, indent=1, default=float))
    g = sc.info["grasp"]
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = g["R"], g["grasp"]
    live_json(r, out, f"scripted {name}", "1khz", T, "scripted")
    print(f"prepared scripted episode {name} for task {r.task.name}: {len(seq)} steps ({sc.P.t:.1f} s sim) -> {out}")
    return 0


def prepare_restore(source, restore_phase, out):
    """The feed of a finished run (source) from its saved <phase>_start.npz on, restored in step 0."""
    source, out = Path(source), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    live = json.loads((source / "live.json").read_text())
    dt = json.loads((source / "episode.json").read_text())["control_dt"]
    snap = np.load(source / "phase_states" / f"{restore_phase}_start.npz")
    b = int(snap["action_index"])
    (out / "restore").mkdir(exist_ok=True)
    for f in (f"{restore_phase}_start.npz", f"{restore_phase}_start_next.npz"):
        shutil.copy(source / "phase_states" / f, out / "restore" / f)
    a = {f: v[b:] for f, v in dict(np.load(source / "actions.npz")).items()}
    ep = shift_episode(json.loads((source / "episode.json").read_text()), b, dt, len(a["grip"]))
    labels = {lab for _, lab, _ in ep["events"]}
    ep["spans"] = [sp for sp in ep.get("spans", []) if sp["start"] in labels and sp["end"] in labels]
    live.update(mode="restore", label=live["label"] + f" restored at {restore_phase}",
                restore=dict(phase=restore_phase, action_index=b, snapshot=f"restore/{restore_phase}_start.npz",
                             snapshot_next=f"restore/{restore_phase}_start_next.npz", source_run=str(source),
                             **({"grasp_rel": snap["grasp_rel"].tolist()} if "grasp_rel" in snap.files else {})))
    np.savez_compressed(out / "actions.npz", **a)
    (out / "episode.json").write_text(json.dumps(ep, indent=1))
    (out / "live.json").write_text(json.dumps(live, indent=1))
    print(f"prepared {live['label']}: {len(a['grip'])} steps from source action {b} -> {out}")
    return 0


def restore_compare(out, live, rep):
    """The restored run against its source run: the same instants (source step = step + offset)."""
    src = Path(live["restore"]["source_run"])
    b = live["restore"]["action_index"]
    A, B = np.load(out / "state_log.npz"), np.load(src / "state_log.npz")
    n = min(len(A["time_s"]), len(B["time_s"]) - b)
    conv = np.degrees if live["unit"] == "deg" else (lambda v: v * 100)
    # from step 1: row 0 is logged before the restore (queued in step 0) takes effect
    d = dict(door=conv(np.abs(A["door_pos"][1:n] - B["door_pos"][b + 1:b + n])),
             tcp_mm=np.linalg.norm(A["tcp_pose"][1:n, :3] - B["tcp_pose"][b + 1:b + n, :3], axis=1) * 1000,
             base_mm=np.linalg.norm(A["base_link_pose"][1:n, :3] - B["base_link_pose"][b + 1:b + n, :3], axis=1) * 1000,
             robot_joint_max=np.abs(A["robot_joint_pos"][1:n] - B["robot_joint_pos"][b + 1:b + n]).max(1))
    src_rep = json.loads((src / "phase_report.json").read_text())
    at = {}
    for p, r in rep["phases"].items():
        if r.get("end_step") is not None:
            e = min(r["end_step"], n) - 2
            at[p] = {k: float(f"{v[e]:.4g}") for k, v in d.items()}
    return dict(rep["restore_check"], steps_compared=n,
                max_diff_over_run={f"{k} ({live['unit'] if k == 'door' else 'rad/m' if k == 'robot_joint_max' else 'mm'})":
                                   float(f"{v.max():.4g}") for k, v in d.items()},
                diff_at_phase_ends=at,
                phase_status=dict(restored={p: r["status"] for p, r in rep["phases"].items()},
                                  source={p: src_rep["phases"][p]["status"] for p in rep["phases"]}))


def finalize(out):
    out = Path(out)
    live = json.loads((out / "live.json").read_text())
    rep = json.loads((out / "phase_live.json").read_text())
    s = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    gz = {c["name"]: c for c in s.get("checks", [])}
    lg = gz.get("load: no [Err] / NaN lines in the gz log, no MUJOCO_LOG", {})
    v = lg.get("value")
    untimed = rep.pop("global_untimed")
    for c in spec.task(live["task"]).task.sanity.untimed:
        if c.metric == "gz_log_problem_lines":    # no summary (gz did not finish): a failing problem
            r = M.evaluate(c, dict(gz_log_problems=v if isinstance(v, list) else ["no gz log check"]))
            untimed.append(ph.from_result(r, what="problem lines in the gz / MuJoCo log (untimed: no regeneration)"))
    rep["global"] = dict(checks=untimed, status="pass" if all(c["pass"] for c in untimed) else "fail",
                         regenerate=False)
    rep["result"] = ph.episode_result(rep["phases"], untimed)
    ok = rep["result"]["failed_phase"] is None
    if live["mode"] == "replay":
        validation.verdict(out)                      # the offline verdict of this run (replay.py verdict)
        this = json.loads((out / "verdict.json").read_text())
        rep["offline_this_run"] = dict(verdict=this["verdict"], reasons=this["reasons"])
        rep["agreement"] = ph.agreement(rep, this)
        ok = ok and this["verdict"] == "accepted"
    elif live["mode"] == "rollout":                  # a policy rollout: the phases are the verdict
        if "restore_check" in rep:                   # a retry from a saved phase state: how well it was restored
            rep["restore_check"]["note"] = "recorded, not gating: the policy, not the source run, drives on"
    elif live["mode"] == "scripted":                 # the scripted episode's own checks (the legacy episode mode)
        rep["episode_checks"] = dict(pass_all=s.get("pass_all") is True, n=len(s.get("checks", [])),
                                     passed=sum(bool(c["pass"]) for c in s.get("checks", [])),
                                     failed=[c["name"] for c in s.get("checks", []) if not c["pass"]])
        ok = ok and rep["episode_checks"]["pass_all"]
    else:
        rc = restore_compare(out, live, rep)
        (out / "restore_check.json").write_text(json.dumps(rc, indent=1))
        rep["restore_check"] = rc
        ok = rc["pass_"]
    rep["accepted"] = bool(ok)
    (out / "phase_report.json").write_text(json.dumps(rep, indent=1))
    line = ph.summary_line(live["label"], rep)
    if live["restore"] is not None and "restore_check" in rep:
        r = rep["restore_check"]
        line += (f" | restore {'PASS' if r['pass_'] else 'FAIL'} {r['vs_source_next_step']['link_pos_m'][0] * 1000:.3f} mm"
                 f" / {r['max_link_rot_deg_vs_next']:.4f} deg")
    (out / "live_summary.txt").write_text(line + "\n")
    print(line)
    return 0 if ok else 1
