"""Simulator entrypoints (run inside revolve-simulator, normally under revolve_common.wrapper):

  cli.py scene-check   --out D [--video]           kitchen scene test (no robot): door/drawer limits, rest
                                                    drift, cup stability (was the earlier `kitchen_test`)
  cli.py robot-smoke   --out D [--video]           mobile Panda smoke test in the kitchen (`kitchen_robot_test`)
  cli.py can-demo      --out D [--video]           the Panda can-on-plate demo
  cli.py episode       --name fridge|drawer --task T --out D [--video]
                       scripted episode (`episode <name>`), fed through the live phase validator: its own
                       checks + per-phase verdicts from specs/tasks/T/task.yaml + phase_states/
  cli.py replay        --plan RUN --attempt A [--task T] --out D [--cameras HZ] [--feed 1khz|30hz] [--no-bag]
                       one AutoMoMa plan attempt: offline verdict + per-phase verdicts + phase_states/
                       + state_log.npz + MCAP bag (bag/, bag.json, bag_check.json)
  cli.py restore-check --source D0 --phase P --out D
                       restore D0's phase_states/<P>_start.npz and run on: state vs the source within
                       task.live tolerances, per-phase verdicts from P on
  cli.py rollout       --start D0 --out D [--seed S] [--restore D1:PHASE] [--budget-s X] [--wait-s X]
                       [--after-success-s X] [--wall-s X] [--client-dir D/client] [--gui]
                       a closed-loop policy rollout (needs revolve-simulator-ros): the robot spawned at the
                       start D0 (a replay-accepted attempt: episode.json, actions.npz row 0, live.json), or
                       restored from D1's phase_states/PHASE_start.npz (a retry); closed_loop_system follows
                       the actions ros_relay.py brings from ROS 2 while the live phase validator (policy
                       mode) judges each phase -> phase_report.json, phase_states/, episode.json (recovery's
                       input: task, seed, plan_run, attempt_id, grasp_id, policy_rollout). The world stays
                       paused until <out>/relay_ready and <client-dir>/runner_ready exist. Budgets: task.yaml
                       `rollout` (sim time); --wall-s bounds the run from the unpause (wall clock).

Exit code = verdict (0 pass). Progress lines `PROGRESS {json}` feed the wrapper's /status.
Paths are absolute or relative to the repo root ($REPO). Outputs go to --out (under runs/).
"""
import argparse
import fcntl
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import spec

SIM = spec.HERE
ALLOW = r"huge value in CTRL at ACTUATOR [0-9]+\. The simulation is unstable\. Time = 0\.0000"
CAMERA_RE = re.compile(r"\n\s*<!-- Fixed scene camera.*?\n    </model>", re.S)
SENSORS_RE = re.compile(r"\n\s*<plugin filename=\"gz-sim-sensors-system\".*?</plugin>", re.S)
EPISODE_CAM = {"fridge": ("2.9 -2.6 2.6 0 0.44 2.33", "1.1"), "drawer": ("-0.6 -1.2 1.7 0 0.414 2.274", "0.9")}


def progress(**kw):
    print("PROGRESS " + json.dumps(kw), flush=True)


def rpath(p):
    p = Path(p)
    return p if p.is_absolute() else spec.ROOT / p


# ------------------------------------------------------------------ models
def _generate(dst, make):
    """Build a generated model dir once (a lock, then an atomic rename: parallel jobs share runs/models)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst.parent / f".{dst.name}.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if not (dst / "model.sdf").is_file():
            tmp = dst.with_name(f".{dst.name}.tmp{os.getpid()}")
            shutil.rmtree(tmp, ignore_errors=True)
            make(tmp)
            shutil.rmtree(dst, ignore_errors=True)
            os.replace(tmp, dst)
    return dst


def models(r):
    """runs/models: mobile_panda with the training cameras (robot.yaml camera_model), and plain/ (no
    cameras: the robot smoke world's model://mobile_panda, the can demo's model://panda)."""
    fd = str(spec.path(r.robot.franka_description))
    gen = [sys.executable, str(SIM / "gen_panda_sdf.py"), fd]
    cam = spec.path(r.robot.camera_model).parent
    _generate(cam, lambda d: (subprocess.run([*gen, str(d), "--mobile"], check=True),
                              subprocess.run([sys.executable, str(SIM / "gen_camera_model.py"), str(d), r.robot.name],
                                             check=True)))
    plain = cam.parent / "plain"
    _generate(plain / "mobile_panda", lambda d: subprocess.run([*gen, str(d), "--mobile"], check=True))
    _generate(plain / "panda", lambda d: subprocess.run([*gen, str(d)], check=True))
    return cam, plain


# ------------------------------------------------------------------ worlds
def world_copy(src, out, video, module=None, robot_uri=None, rtf=None, camera=None):
    """<out>/world.sdf: src with the scene camera kept (video) or dropped with the Sensors system (render
    off), the Python system swapped for `module`, the robot include pointed at robot_uri, the RTF set."""
    w = Path(src).read_text()
    if video:
        if camera:
            w = w.replace("<pose>2.75 -2.75 2.6 0 0.42 2.40</pose>", f"<pose>{camera[0]}</pose>")
            w = w.replace("<horizontal_fov>1.25</horizontal_fov>", f"<horizontal_fov>{camera[1]}</horizontal_fov>")
    else:
        w, n = CAMERA_RE.subn("", w)
        w, m = SENSORS_RE.subn("", w)
        if n != 1 or m != 1:
            raise SystemExit(f"{src}: scene camera ({n}) / Sensors system ({m}) not found exactly once")
    if module:
        w, n = re.subn(r"<module_name>[a-z_]+</module_name>", f"<module_name>{module}</module_name>", w)
        assert n == 1, f"{src}: {n} python systems"
    if robot_uri:
        w = w.replace("<uri>model://mobile_panda</uri>", f"<uri>{robot_uri}</uri>")
    if rtf is not None:
        w, n = re.subn(r"<real_time_factor>[0-9.]+</real_time_factor>", f"<real_time_factor>{rtf}</real_time_factor>", w)
        assert n == 1
    (Path(out) / "world.sdf").write_text(w)
    return Path(out) / "world.sdf"


# ------------------------------------------------------------------ one gz run
def run_gz(out, resource, driver, env=None, render=False, wait_level=False, side=None, gui=False):
    """gz sim (server, headless) on <out>/world.sdf with cwd out (MUJOCO_LOG.TXT lands there), the driver
    script beside it; side: an extra process (camera capture) for the run; gui: also a `gz sim -g` client on
    $DISPLAY (log <out>/gz_gui.log), never part of the verdict: closing its window leaves the server running.
    -> the driver's exit code."""
    out = Path(out)
    e = dict(os.environ, GZ_SIM_RESOURCE_PATH=resource, KITCHEN_ASSETS="partnet",
             GZ_SIM_SYSTEM_PLUGIN_PATH=f"{SIM}:{os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', '')}",
             GZ_PARTITION=f"revolve_{os.getpid()}_{random.randrange(1 << 30)}",
             GZ_MUJOCO_NOSLIP_ITERATIONS=os.environ.get("GZ_MUJOCO_NOSLIP_ITERATIONS", "10"),
             **(env or {}))
    if render:      # the GPU EGL vendor file (no NVIDIA one in the image), Mesa as fallback; NVENC for the MP4
        e.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                     f"{SIM.parent / 'egl' / '10_nvidia.json'}:/usr/share/glvnd/egl_vendor.d/50_mesa.json")
    for f in ("MUJOCO_LOG.TXT", "gz_sim.log"):
        (out / f).unlink(missing_ok=True)
    log = open(out / "gz_sim.log", "w")
    gz = subprocess.Popen(["gz", "sim", "-s", "--headless-rendering", "-v", "3", "world.sdf"], cwd=out, env=e,
                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    sp = subprocess.Popen(side, env=e) if side else None
    gl = open(out / "gz_gui.log", "w") if gui else None
    ge = dict(e, QT_QPA_PLATFORM="xcb", __NV_PRIME_RENDER_OFFLOAD="1", __GLX_VENDOR_LIBRARY_NAME="nvidia")  # GLX on the GPU
    gp = subprocess.Popen(["gz", "sim", "-g", "-v", "3"], cwd=out, env=ge, stdout=gl, stderr=subprocess.STDOUT,
                          start_new_session=True) if gui else None
    try:
        if wait_level:   # an early unpause runs steps without the Python system (seen on Modal): wait for the level
            for _ in range(600):
                if "Loaded level [default]" in (out / "gz_sim.log").read_text(errors="replace") or gz.poll() is not None:
                    break
                time.sleep(0.1)
        with open(out / "driver.log", "w") as dl:
            p = subprocess.Popen(driver, cwd=out, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                dl.write(line)
                if line.startswith(("LIVE", "[skill] LIVE", "DONE", "Timed out")) or " PHASE " in line \
                        or line.startswith(("PASS", "FAIL")) or "PASS (" in line or "FAIL (" in line:
                    print(line.rstrip(), flush=True)
                if " PHASE " in line:
                    m = re.search(r"PHASE (\w+) (PASS|FAIL)", line)
                    if m:
                        progress(phase=m.group(1), live=m.group(2))
            rc = p.wait()
    finally:
        for proc in (gz, sp, gp):
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM) if proc is not sp else proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL) if proc is not sp else proc.kill()
                    proc.wait()
        log.close()
        if gl:
            gl.close()
    if sp is not None and sp.returncode not in (0, -signal.SIGTERM):
        rc = rc or sp.returncode
    return rc


def kitchen_driver(out, status, label, timeout, video=None, allow=True):
    return [sys.executable, str(SIM / "kitchen_test.py"), "--out", str(out), "--log", str(Path(out) / "gz_sim.log"),
            "--status", status, "--label", label, "--timeout", str(timeout),
            *(["--allow", ALLOW] if allow else []), *(["--video", str(Path(out) / "video.mp4")] if video else [])]


def summary(out):
    p = Path(out) / "summary.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def legacy_result(out, rc, label):
    s = summary(out)
    n, ok = len(s.get("checks", [])), sum(bool(c["pass"]) for c in s.get("checks", []))
    res = dict(label=label, pass_=rc == 0, checks_passed=ok, checks=n, wall_s=s.get("wall_s"),
               failed=[c["name"] for c in s.get("checks", []) if not c["pass"]])
    (Path(out) / "result_sim.json").write_text(json.dumps(res, indent=1))
    print(f"{label} {'PASS' if rc == 0 else 'FAIL'} {ok}/{n} checks")
    return rc


# ------------------------------------------------------------------ entrypoints
def task_of(a):
    return spec.task(a.task)


def scene_check(a):
    out = rpath(a.out)
    out.mkdir(parents=True, exist_ok=True)
    r = task_of(a)
    world_copy(spec.WORLDS / "kitchen_partnet.sdf", out, a.video)
    progress(stage="gz", what="scene-check")
    rc = run_gz(out, spec.resource_path(r), kitchen_driver(out, "/kitchen_test/status", "KITCHEN_TEST", 300, a.video,
                                                           allow=False),
                env=dict(KITCHEN_TEST_OUT=str(out / "kitchen_test.json")), render=a.video)
    return legacy_result(out, rc, "KITCHEN_TEST")


def robot_smoke(a):
    out = rpath(a.out)
    out.mkdir(parents=True, exist_ok=True)
    r = task_of(a)
    _, plain = models(r)
    world_copy(spec.WORLDS / "kitchen_partnet_robot.sdf", out, a.video)
    progress(stage="gz", what="robot-smoke")
    rc = run_gz(out, f"{plain}:{spec.resource_path(r)}",
                kitchen_driver(out, "/mobile_panda/status", "MOBILE_PANDA_SMOKE", 600, a.video),
                env=dict(KITCHEN_TEST_OUT=str(out / "kitchen_test.json")), render=a.video)
    return legacy_result(out, rc, "MOBILE_PANDA_SMOKE")


def can_demo(a):
    out = rpath(a.out)
    out.mkdir(parents=True, exist_ok=True)
    r = task_of(a)
    _, plain = models(r)
    w = (spec.WORLDS / "can_on_plate.sdf").read_text()
    if not a.video:
        w = SENSORS_RE.sub("", CAMERA_RE.sub("", w))
    (out / "world.sdf").write_text(w)
    progress(stage="gz", what="can-demo")
    rc = run_gz(out, str(plain), [sys.executable, str(SIM / "pick_place.py"),
                                  *(["--video", str(out / "video.mp4")] if a.video else [])], render=a.video)
    log = (out / "driver.log").read_text()
    m = re.search(r"DONE (SUCCESS|FAILED).*", log)
    d = re.search(r"([0-9.]+) cm from", m.group(0)) if m else None
    res = dict(label="CAN_DEMO", pass_=rc == 0, done=m.group(0) if m else None,
               distance_cm=float(d.group(1)) if d else None)
    (out / "result_sim.json").write_text(json.dumps(res, indent=1))
    print(f"CAN_DEMO {'PASS' if rc == 0 else 'FAIL'}: {res['done']}")
    return rc


def phase_run(out, r, label, video=False, cameras=0, camera=None, timeout=840):
    """The world copy of the task's scene with phase_replay_system on <out>/actions.npz + episode.json."""
    import live_validate
    cam_model, _ = models(r)
    render = bool(video or cameras)
    world_copy(spec.path(r.scene.world), out, render, module="phase_replay_system",
               robot_uri=str(cam_model), rtf=spec.SIM["real_time_factor"], camera=camera)
    if cameras and not video:     # the training cameras render, the scene camera does not
        w = (out / "world.sdf").read_text()
        (out / "world.sdf").write_text(CAMERA_RE.sub("", w))
    side = [sys.executable, str(SIM / "capture.py"), str(cam_model), str(out), str(cameras)] if cameras else None
    progress(stage="gz", what=label)
    # model://mobile_panda/meshes of the generated model resolve through its parent (as output/models did)
    rc = run_gz(out, f"{cam_model.parent}:{spec.resource_path(r)}", kitchen_driver(out, "/mobile_panda_skill/status", label, timeout,
                                                           video and not cameras),
                env=dict(REPLAY_DIR=str(out), REPLAY_RTF=str(spec.SIM["real_time_factor"]),
                         KITCHEN_TEST_OUT=str(out / "gz_result.json"), SKILL="automoma_replay",
                         EPISODE=label.lower()),
                render=render, wait_level=True, side=side)
    if not (out / "phase_live.json").is_file():
        print(f"no phase_live.json (driver rc {rc})")
        return 3
    progress(stage="finalize")
    return live_validate.finalize(out)


EPISODE_TASK = {"fridge": "open_fridge_door", "drawer": "open_drawer"}


def episode(a):
    import live_validate
    out = rpath(a.out)
    if out.exists():
        shutil.rmtree(out)
    r = spec.task(a.task or EPISODE_TASK[a.name])
    models(r)
    os.environ["GZ_SIM_RESOURCE_PATH"] = spec.resource_path(r)     # kitchen_episodes reads the manifests
    live_validate.prepare_scripted(a.name, out, r.task.name)
    return phase_run(out, r, f"EPISODE_{a.name.upper()}", video=a.video, camera=EPISODE_CAM[a.name])


def replay(a):
    import live_validate
    out = rpath(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    t0 = time.time()
    plan = rpath(a.plan)
    r = spec.task(a.task or json.loads((plan / "handoff.json").read_text())["plugins"]["task"])
    cam, _ = models(r)
    os.environ["SIM_ROBOT_MODEL"] = str(cam / "model.sdf")
    os.environ["GZ_SIM_RESOURCE_PATH"] = f"{cam.parent}:{spec.resource_path(r)}"   # the clearance precheck reads models
    progress(stage="prepare", plan=plan.name, attempt=a.attempt)
    if live_validate.prepare_plan(plan, a.attempt, out, r.task.name, a.feed) != 0:
        print("spawn in collision: rejected before gz")
        return 1
    rc = phase_run(out, r, "REPLAY", cameras=a.cameras, timeout=600)
    if rc in (0, 1) and not a.no_bag:
        progress(stage="bag")
        subprocess.run([sys.executable, str(SIM / "bag.py"), "write", str(out)], check=True)
        chk = subprocess.run([sys.executable, str(SIM / "bag.py"), "check", str(out / "bag"), str(out)],
                             capture_output=True, text=True)
        (out / "bag_check.json").write_text(chk.stdout)
        if chk.returncode:
            print(f"bag check FAILED: {chk.stdout[-500:]}")
            rc = 4
    print(f"replay done in {time.time() - t0:.0f} s wall, rc {rc}")
    return rc


def restore_check(a):
    import live_validate
    out = rpath(a.out)
    if out.exists():
        shutil.rmtree(out)
    src = rpath(a.source)
    live_validate.prepare_restore(src, a.phase, out)
    r = spec.task(json.loads((src / "live.json").read_text())["task"])
    return phase_run(out, r, "RESTORE", timeout=600)


ROLLOUT_WAIT_CLIENT_S = 300      # wall clock for the policy client (server + Rosetta runner) to come up


def rollout_spec(start, out, r, seed=None, restore=None, budget_s=None, wait_s=None, after_success_s=None):
    """<out>/closed_loop.json, live.json (policy mode) and episode.json of a rollout from the start dir (or a
    restored phase state: restore = (source rollout dir, phase)). Ported from the legacy closed_loop.py spec
    + policy_episode.py make (grasp pose / segment); thresholds and budgets from the task spec."""
    import numpy as np
    import live_validate
    import mobile_panda as mp
    import panda_plan as pp
    import sdf
    start, out = Path(start), Path(out)
    ep = json.loads((start / "episode.json").read_text())
    slive = json.loads((start / "live.json").read_text())
    t, ro = r.task, r.task.rollout
    a = mp.ActionSeq.load(start / "actions.npz")[0]
    spawn = dict(base=a.base.tolist(), arm=a.arm.tolist(), grip=float(a.grip))
    rest = None
    if restore:
        src, phase = Path(restore[0]), restore[1]
        (out / "restore").mkdir(parents=True, exist_ok=True)
        for f in (f"{phase}_start.npz", f"{phase}_start_next.npz"):
            shutil.copy(src / "phase_states" / f, out / "restore" / f)
        snap = np.load(out / "restore" / f"{phase}_start.npz")
        spawn = dict(base=snap["action_base"].tolist(), arm=snap["action_arm"].tolist(), grip=float(snap["action_grip"]))
        rest = dict(phase=phase, action_index=int(snap["action_index"]), snapshot=f"restore/{phase}_start.npz",
                    snapshot_next=f"restore/{phase}_start_next.npz", source_run=str(src),
                    **({"grasp_rel": snap["grasp_rel"].tolist()} if "grasp_rel" in snap.files else {}))
    _, part, unit, si, _ = spec.hold(r)
    th = dict(spec.legacy_thresholds(r), grasp_half_width_m=ep["thresholds"]["grasp_half_width_m"])
    key = f"{part}_open_min_{unit}"
    pb = t.boundaries.policy
    window = spec.cond(r, "grasp", "finger_contacts").window_s
    success = dict(joint_min=si(th[key]), hold_s=th["hold_s"], threshold=th[key], unit=unit,
                   check=dict(slip_max_m=th["handle_slip_max_m"], half_width_m=th["grasp_half_width_m"],
                              width_tol_m=th["grasp_width_tol_m"], closed_max_m=pb.grip_closed_max_m,
                              settle_s=pb.grasp_settle_s, window_s=window, moved_eps=si(pb.joint_moved_eps)))
    cam, _ = models(r)
    robot = sdf.Model(cam / "model.sdf")
    base = np.array(r.robot.gz_base_limits, float)
    contract = yaml_load(spec.SPECS / "robots" / r.robot.name / r.robot.contract)
    cl = dict(plan_run=ep["plan_run"], attempt_id=ep["attempt_id"], grasp_id=ep["grasp_id"], task=t.name,
              prompt=t.prompt, start_dir=str(start), scene=r.scene.name, world=r.scene.world,
              object_joint=ep["object_joint"], object_handle=ep["object_handle"], joint_topic=ep["joint_topic"],
              hold=ep["hold"], thresholds=th, contact_rate_hz=ep["contact_rate_hz"], success=success, spawn=spawn,
              fps=contract["fps"], budget_s=float(budget_s if budget_s is not None else ro.budget_s),
              after_success_s=ro.after_success_s if after_success_s is None else after_success_s,
              wait_s=float(wait_s if wait_s is not None else ro.wait_s),
              # action order: base x, y, yaw, panda_joint1-7, gripper; velocity: the first 10
              limits=dict(lower=[*-base, *pp.LOWER, mp.GRIP_CLOSED], upper=[*base, *pp.UPPER, mp.GRIP_OPEN],
                          velocity=[mp.BASE_V_MAX, mp.BASE_V_MAX, mp.BASE_W_MAX] +
                                   [robot.joints[f"panda_joint{i}"]["velocity"] for i in range(1, 8)]),
              subtasks=["closed_loop"], event_subtask=["closed_loop"], restore=rest)
    (out / "closed_loop.json").write_text(json.dumps(cl, indent=1, default=float))
    grasp = np.asarray(slive["grasp_pose_world"], float)
    zs = (t.planner.get("grasp") or {}).get("z")     # a vertical bar: any trained grasp height (policy_episode.py)
    p0 = grasp[:3, 3]
    seg = [[*p0[:2].tolist(), min(zs)], [*p0[:2].tolist(), max(zs)]] if zs else None
    label = f"rollout seed {seed} from {start.name}" + (f" restored at {restore[1]}" if restore else "")
    live = live_validate.live_json(r, out, label, "closed_loop", grasp, "rollout", start=str(start), seed=seed)
    live.update(policy=dict(success=success, grasp_segment_world=seg), restore=rest)
    (out / "live.json").write_text(json.dumps(live, indent=1))
    # recovery's input (components/recovery/recover.py source()): task, seed, the start's plan ids, policy_rollout
    (out / "episode.json").write_text(json.dumps(dict(
        task=t.name, prompt=t.prompt, seed=seed, control_dt=mp.CONTROL_DT, plan_run=ep["plan_run"],
        attempt_id=ep["attempt_id"], grasp_id=ep["grasp_id"], object_joint=ep["object_joint"],
        hold=ep["hold"], thresholds=th, policy_rollout=dict(start=str(start), restored_from=rest,
                                                             budget_s=cl["budget_s"], wait_s=cl["wait_s"],
                                                             after_success_s=cl["after_success_s"])), indent=1))
    return cl


def yaml_load(p):
    import yaml
    return yaml.safe_load(Path(p).read_text())


def rollout(a):
    import live_validate
    out = rpath(a.out)
    out.mkdir(parents=True, exist_ok=True)     # not emptied: the host made it, the client writes beside it
    client = rpath(a.client_dir) if a.client_dir else out / "client"
    start = rpath(a.start)
    r = spec.task(json.loads((start / "episode.json").read_text())["task"])
    restore = None
    if a.restore:
        d, _, ph_ = a.restore.rpartition(":")
        restore = (rpath(d), ph_)
    progress(stage="prepare", seed=a.seed, restore=a.restore)
    cl = rollout_spec(start, out, r, a.seed, restore, a.budget_s, a.wait_s, a.after_success_s)
    cam, _ = models(r)
    world_copy(spec.path(r.scene.world), out, True, module="closed_loop_system", robot_uri=str(cam),
               rtf=r.task.rollout.real_time_factor)
    w = (out / "world.sdf").read_text()       # the training cameras render, the scene camera does not
    (out / "world.sdf").write_text(CAMERA_RE.sub("", w))
    contract = spec.SPECS / "robots" / r.robot.name / r.robot.contract
    relay = ["bash", "-c", 'source /opt/ros/jazzy/setup.bash && exec python3 "$@"', "relay",
             str(SIM / "ros_relay.py"), str(out), str(contract), str(cam)]
    ready = f"{out}/relay_ready {client}/runner_ready"
    driver = ["bash", "-c", f'for i in $(seq {ROLLOUT_WAIT_CLIENT_S * 10}); do ok=1; for f in {ready}; do [[ -f $f ]] || ok=0; '
              f'done; (( ok )) && break; sleep 0.1; done; (( ok )) || {{ echo "Timed out waiting for {ready}"; exit 4; }}; '
              f'echo "client ready: unpausing"; exec "$@"', "driver",
              *kitchen_driver(out, "/mobile_panda_skill/status", "ROLLOUT", a.wall_s)]
    progress(stage="gz", what="rollout", budget_s=cl["budget_s"], wait_s=cl["wait_s"])
    rc = run_gz(out, f"{cam.parent}:{spec.resource_path(r)}", driver,
                env=dict(REPLAY_DIR=str(out), CL_DIR=str(out), REPLAY_RTF=str(r.task.rollout.real_time_factor),
                         KITCHEN_TEST_OUT=str(out / "gz_result.json"), SKILL="closed_loop", EPISODE="closed_loop"),
                render=True, wait_level=True, side=relay, gui=a.gui)
    if not (out / "phase_live.json").is_file():
        print(f"no phase_live.json (driver rc {rc})")
        return 3
    progress(stage="finalize")
    return live_validate.finalize(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cmds = {"scene-check": scene_check, "robot-smoke": robot_smoke, "can-demo": can_demo, "episode": episode,
            "replay": replay, "restore-check": restore_check, "rollout": rollout}
    p = {k: sub.add_parser(k) for k in cmds}
    for k in p.values():
        k.add_argument("--out", required=True)
    for k in ("scene-check", "robot-smoke", "can-demo", "episode"):
        p[k].add_argument("--video", action="store_true", help="render the scene camera into <out>/video.mp4 (GPU)")
        p[k].add_argument("--task", default=None if k == "episode" else "open_fridge_door",
                          help="task spec (its robot and scene); episode default: the episode's task")
    p["episode"].add_argument("--name", required=True, choices=sorted(EPISODE_CAM))
    p["replay"].add_argument("--plan", required=True)
    p["replay"].add_argument("--attempt", required=True, type=int)
    p["replay"].add_argument("--task", default=None, help="default: the plan's (handoff.json)")
    p["replay"].add_argument("--cameras", type=int, default=0, help="training camera rate Hz (0 = render off)")
    p["replay"].add_argument("--feed", default="1khz", choices=("1khz", "30hz"))
    p["replay"].add_argument("--no-bag", action="store_true")
    p["restore-check"].add_argument("--source", required=True)
    p["restore-check"].add_argument("--phase", required=True)
    p["rollout"].add_argument("--start", required=True)
    p["rollout"].add_argument("--seed", type=int, default=None)
    p["rollout"].add_argument("--restore", default=None, help="SOURCE_ROLLOUT_DIR:PHASE (a retry)")
    p["rollout"].add_argument("--budget-s", type=float, default=None, help="default: task.yaml rollout.budget_s")
    p["rollout"].add_argument("--wait-s", type=float, default=None, help="default: task.yaml rollout.wait_s")
    p["rollout"].add_argument("--after-success-s", type=float, default=None)
    p["rollout"].add_argument("--wall-s", type=float, default=3600.0, help="wall-clock limit from the unpause")
    p["rollout"].add_argument("--client-dir", default=None, help="default: <out>/client")
    p["rollout"].add_argument("--gui", action="store_true", help="also show the gz GUI on $DISPLAY (launch.py gui)")
    a = ap.parse_args(argv)
    return cmds[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
