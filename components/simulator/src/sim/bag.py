"""MCAP rosbag2 bag of one replay: simulator state and actions, plus the training cameras' frames
when the replay is a camera replay (cameras on: capture.json and <camera>/*.png).
rosbags (+ Pillow for JPEG), no ROS install; runs in revolve-simulator (cli.py replay). The robot is
$REVOLVE_ROBOT's spec: /action joint names = its joints.action, the final state's joint scope = its gz model.

  bag.py write <replay dir>
      <replay>/state_log.npz + verdict.json -> <replay>/bag/ (metadata.yaml + bag.mcap, zstd
      chunks) and <replay>/bag.json (files, size, topics, message counts, write wall time).
      A camera replay's PNGs inside the state log's window go in as JPEG q90 CompressedImage.
  bag.py check <bag dir> [<replay dir>]
      opens the bag with the rosbags reader and checks it against the replay (default: the
      bag's parent dir): no camera data (a camera replay: one image topic per camera, frames on
      the camera grid), one message per 1 ms step with stamps n x 1 ms, content =
      state_log.npz, last joint state = final_state.npz, metadata = verdict.json, size recorded.
      JSON on stdout; exit code 0 = every check passes.

The task object's joint (the fridge door hinge or the drawer slide, state log door_*) goes on the
task's topic, episode.json joint_topic: /fridge/joint_states or /drawer/joint_states.

Stamps are sim time from the spawn: header stamp = log time = n x control_dt for step n. Row n
of state_log.npz is the state at the start of step n and the action applied during it, so the
bag's last joint state is the sim one step before final_state.npz (the check uses that step).
An image's header stamp is its gz render time, on the same clock: the frame shows state log row
stamp / control_dt. Its log time is its tick k x int(1e9 / Hz) ns, up to 1 ms before the render (gz
renders at the first 1 ms step at or after k / Hz), so a resampler that ticks at Hz from the bag start
on log time (Rosetta's porter) pairs tick k with frame k, not frame k - 1.
"""
import io
import json
import shutil
import sys
import time
from importlib.metadata import version
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import CompressionFormat, CompressionMode, Reader, StoragePlugin, Writer
from rosbags.typesys import Stores, get_typestore
from ruamel.yaml import YAML

JS, PS, STR = "sensor_msgs/msg/JointState", "geometry_msgs/msg/PoseStamped", "std_msgs/msg/String"
CI = "sensor_msgs/msg/CompressedImage"
JPEG_QUALITY = 90
PROMPT = "lerobot.operator_prompt"   # Rosetta reads the task string from this custom_data key
ON_CHANGE = {"/subtask": STR, "/event": STR, "/contacts": STR}
CUSTOM = ("plan_run", "attempt_id", "grasp_id", "verdict", "failed_checks", "steps", "control_dt_s",
          "verdict_json")
EULER_TOL = 1e-9   # final pos vs last pos + final vel x dt (MuJoCo's semi-implicit Euler step)
TS = get_typestore(Stores.ROS2_JAZZY)


def robot():
    """The robot spec ($REVOLVE_ROBOT): /action names (joints.action) and the gz joint scope (model)."""
    import os
    import spec
    from revolve_common import specs
    return specs.load(spec.SPECS / "robots" / os.environ[specs.ROBOT_ENV] / "robot.yaml")


def per_step(obj):
    """Per-step topics; obj = the object joint's topic."""
    return {"/joint_states": JS, "/action": JS, obj: JS, "/base_pose": PS, "/tcp_pose": PS}


def episode(replay):
    ep = Path(replay) / "episode.json"
    return json.loads(ep.read_text()) if ep.is_file() else {}


def object_topic(replay):
    """episode.json joint_topic; replays from before the drawer task have none and are the fridge's."""
    return episode(replay).get("joint_topic", "/fridge/joint_states")


def capture(replay):
    """capture.json of a camera replay, else None."""
    p = Path(replay) / "capture.json"
    return json.loads(p.read_text()) if p.is_file() else None


def image_topic(camera):
    return f"/cameras/{camera.removesuffix('_cam')}/image/compressed"


def camera_hz(cap, camera):
    return min(cap["rate_hz"], cap["cameras"][camera]["camera_hz"])


def tick_ns(ns, hz):
    """Log time of a frame rendered at ns: its 1 / hz tick on the grid of Rosetta's porter and ROS 2
    rates, k x int(1e9 / hz) (rosetta.frames.resample.step_ns)."""
    return round(ns * hz / 1e9) * int(1e9 / hz)


def frames(replay, end_ns):
    """camera -> sorted [(stamp ns, png)] of a camera replay's PNGs with stamp < end_ns ({} for a
    plain replay). A PNG is named by its gz stamp in s, 3 decimals: whole physics steps."""
    cap = capture(replay)
    fs = {c: sorted((round(float(p.stem) * 1e9), p) for p in (Path(replay) / c).glob("*.png"))
          for c in (cap["cameras"] if cap else [])}
    return {c: [(ns, p) for ns, p in f if ns < end_ns] for c, f in fs.items()}


def jpeg(png):
    from PIL import Image   # in automoma-bags since the camera bags; an older image: docker rmi automoma-bags
    b = io.BytesIO()
    with Image.open(png) as im:
        im.convert("RGB").save(b, "JPEG", quality=JPEG_QUALITY)
    return b.getvalue()


def content(s, obj):
    """Per-step arrays of each topic, from state_log.npz (the writer's source, the check's reference)."""
    n = len(s["time_s"])
    sub, ev = s["subtask_labels"][s["subtask"]], s["event_labels"][s["event"]]
    contacts = np.stack([s["contact_left_finger_handle"], s["contact_right_finger_handle"],
                         s["contact_other_robot"], s["contact_robot_max_depth_m"]], 1).astype(float)
    return {
        "/joint_states": (s["robot_joint_pos"], s["robot_joint_vel"], s["robot_joint_effort"]),
        # gripper: per-finger opening target, no feed-forward velocity, max squeeze per finger as
        # effort; base and arm targets set no effort (NaN)
        "/action": (np.hstack([s["action_base"], s["action_arm"], s["action_grip"][:, None]]),
                    np.hstack([s["action_base_vel"], s["action_arm_vel"], np.zeros((n, 1))]),
                    np.hstack([np.full((n, s["action_base"].shape[1] + s["action_arm"].shape[1]), np.nan),
                               s["action_grip_force"][:, None]])),
        obj: (s["door_pos"][:, None], s["door_vel"][:, None], np.zeros((n, 0))),
        "/base_pose": s["base_link_pose"], "/tcp_pose": s["tcp_pose"],
        "/subtask": [json.dumps({"subtask": str(x)}) for x in sub],
        "/event": [json.dumps({"event": str(x)}) for x in ev],
        "/contacts": [json.dumps({"left_finger_handle": bool(c[0]), "right_finger_handle": bool(c[1]),
                                  "other_robot": int(c[2]), "robot_max_depth_m": float(c[3])})
                      for c in contacts],
    }


def write(replay):
    replay = Path(replay)
    t0 = time.time()
    s = np.load(replay / "state_log.npz")
    v = json.loads((replay / "verdict.json").read_text())
    n, dt = len(s["time_s"]), float(s["control_dt"])
    dt_ns = round(dt * 1e9)
    if not np.allclose(s["time_s"], np.arange(n) * dt, rtol=0, atol=1e-9):
        sys.exit("state_log time_s is not n x control_dt")
    if (str(s["plan_run"]), int(s["attempt_id"])) != (v["plan_run"], v["attempt_id"]):
        sys.exit("state_log and verdict.json name different attempts")
    obj = object_topic(replay)
    PER_STEP, c = per_step(obj), content(s, obj)
    cap, fr = capture(replay), frames(replay, n * dt_ns)
    IMAGES = {image_topic(cam): CI for cam in fr}
    with ThreadPoolExecutor(4) as pool:      # Pillow releases the GIL while it codes
        jpg = {cam: list(pool.map(jpeg, [p for _, p in f])) for cam, f in fr.items()}
    images = sorted((tick_ns(ns, camera_hz(cap, cam)), ns, cam, b) for cam, f in fr.items()
                    for (ns, _), b in zip(f, jpg[cam]))
    prompt = episode(replay).get("prompt")
    T = TS.types
    Time, Header = T["builtin_interfaces/msg/Time"], T["std_msgs/msg/Header"]
    JointState, String = T[JS], T[STR]
    Pose, Point, Quat = T["geometry_msgs/msg/Pose"], T["geometry_msgs/msg/Point"], T["geometry_msgs/msg/Quaternion"]
    names = {"/joint_states": [str(x) for x in s["robot_joint_names"]], "/action": robot().joints.action,
             obj: [str(s["door_joint"]).split("::")[-1]]}
    changed = {t: [0] + [k for k in range(1, n) if c[t][k] != c[t][k - 1]] for t in ON_CHANGE}

    out = replay / "bag"
    shutil.rmtree(out, ignore_errors=True)   # generated; the Writer never overwrites
    (replay / "bag.json").unlink(missing_ok=True)
    w = Writer(out, version=9, storage_plugin=StoragePlugin.MCAP)
    w.set_compression(CompressionMode.STORAGE, CompressionFormat.ZSTD)   # MCAP chunk compression
    failed = [x["name"] for x in v["checks"] if not x["pass"]]
    custom = dict(plan_run=v["plan_run"], attempt_id=v["attempt_id"], grasp_id=v["grasp_id"],
                  verdict=v["verdict"], failed_checks=json.dumps(failed), steps=n, control_dt_s=dt,
                  verdict_json=json.dumps(v))
    for k in CUSTOM:
        w.set_custom_data(k, str(custom[k]))
    if prompt:
        w.set_custom_data(PROMPT, prompt)
    counts = dict.fromkeys([*PER_STEP, *ON_CHANGE, *IMAGES], 0)
    with w:
        conn = {t: w.add_connection(t, ty, typestore=TS) for t, ty in {**PER_STEP, **ON_CHANGE, **IMAGES}.items()}

        def put(topic, msg, ns):
            w.write(conn[topic], ns, TS.serialize_cdr(msg, msg.__msgtype__))
            counts[topic] += 1

        on_change = sorted((k, t) for t in ON_CHANGE for k in changed[t])
        j = i = 0
        for k in range(n):
            ns = k * dt_ns
            stamp = Time(sec=ns // 10**9, nanosec=ns % 10**9)
            for t in ("/joint_states", "/action", obj):
                p, vel, eff = (a[k] for a in c[t])
                put(t, JointState(header=Header(stamp=stamp, frame_id=""), name=names[t],
                                  position=p, velocity=vel, effort=eff), ns)
            for t in ("/base_pose", "/tcp_pose"):
                x, y, z, qw, qx, qy, qz = (float(a) for a in c[t][k])
                put(t, T[PS](header=Header(stamp=stamp, frame_id="world"),
                             pose=Pose(position=Point(x=x, y=y, z=z), orientation=Quat(x=qx, y=qy, z=qz, w=qw))), ns)
            while j < len(on_change) and on_change[j][0] == k:
                put(on_change[j][1], String(data=c[on_change[j][1]][k]), ns)
                j += 1
            while i < len(images) and images[i][0] < ns + dt_ns:
                t_log, t_ns, cam, b = images[i]
                put(image_topic(cam), T[CI](header=Header(stamp=Time(sec=t_ns // 10**9, nanosec=t_ns % 10**9),
                                                          frame_id=cam), format="jpeg",
                                            data=np.frombuffer(b, np.uint8)), t_log)
                i += 1
    wall = time.time() - t0
    files = {p.name: p.stat().st_size for p in sorted(out.iterdir())}
    info = dict(bag=out.name, storage="mcap", compression="zstd per MCAP chunk (rosbag2 storage mode)",
                rosbags=version("rosbags"), files=files, size_bytes=sum(files.values()),
                size_mb=round(sum(files.values()) / 1e6, 2), steps=n, control_dt_s=dt,
                topics={t: dict(type=ty, count=counts[t]) for t, ty in {**PER_STEP, **ON_CHANGE, **IMAGES}.items()},
                messages=sum(counts.values()), write_wall_s=round(wall, 2),
                messages_per_s=round(sum(counts.values()) / wall), custom_data_keys=list(CUSTOM) + [PROMPT] * bool(prompt))
    if images:
        info["images"] = dict(format="jpeg", quality=JPEG_QUALITY, bytes=sum(len(b) for *_, b in images))
    (replay / "bag.json").write_text(json.dumps(info, indent=1))
    print(f"BAG {v['plan_run']}/{v['attempt_id']} {v['verdict']}: {info['messages']} messages, "
          f"{info['size_mb']} MB, {info['write_wall_s']} s -> {out}")


def mcap_metadata(path):
    """name -> {key: value} of every MCAP Metadata record (op 0x0C), by a linear record scan."""
    b = path.read_bytes()

    def string(o):
        k = int.from_bytes(b[o:o + 4], "little")
        return b[o + 4:o + 4 + k].decode(), o + 4 + k

    out, o = {}, 8
    while o < len(b) - 8:
        op, ln = b[o], int.from_bytes(b[o + 1:o + 9], "little")
        if op == 0x0C:
            name, p = string(o + 9)
            end, p, kv = p + 4 + int.from_bytes(b[p:p + 4], "little"), p + 4, {}
            while p < end:
                key, p = string(p)
                kv[key], p = string(p)
            out[name] = kv
        o += 9 + ln
    return out


def check(bag, replay=None):
    bag = Path(bag)
    replay = Path(replay) if replay else bag.parent
    s, f = np.load(replay / "state_log.npz"), np.load(replay / "final_state.npz")
    v = json.loads((replay / "verdict.json").read_text())
    n, dt = len(s["time_s"]), float(s["control_dt"])
    dt_ns = round(dt * 1e9)
    obj = object_topic(replay)
    PER_STEP, ref = per_step(obj), content(s, obj)
    cap = capture(replay)
    IMAGES = {image_topic(cam): cam for cam in (cap["cameras"] if cap else [])}
    checks = []

    def add(name, ok, value, note):
        checks.append({"name": name, "pass": bool(ok), "value": value, "note": note})

    got = {t: [] for t in {**PER_STEP, **ON_CHANGE, **IMAGES}}
    stamps = {t: [] for t in got}
    with Reader(bag) as r:
        topics = {t: dict(type=i.msgtype, count=i.msgcount) for t, i in r.topics.items()}
        for con, ns, raw in r.messages():
            m = TS.deserialize_cdr(raw, con.msgtype)
            h = getattr(m, "header", None)
            stamps[con.topic].append((ns, h.stamp.sec * 10**9 + h.stamp.nanosec if h else ns))
            if con.msgtype == JS:
                got[con.topic].append((m.position, m.velocity, m.effort))
            elif con.msgtype == PS:
                p, q = m.pose.position, m.pose.orientation
                got[con.topic].append((p.x, p.y, p.z, q.w, q.x, q.y, q.z))
            elif con.msgtype == CI:
                got[con.topic].append((m.header.frame_id, m.format, bytes(m.data[:2]) + bytes(m.data[-2:])))
            else:
                got[con.topic].append((ns // dt_ns, m.data))
    meta = YAML(typ="safe").load((bag / "metadata.yaml").read_text())["rosbag2_bagfile_information"]
    cd = meta.get("custom_data") or {}
    add("opens_with_rosbags", True, dict(storage=meta["storage_identifier"], files=meta["relative_file_paths"],
                                         rosbags=version("rosbags")), "rosbags Reader opened and read every message")
    add("mcap_storage", meta["storage_identifier"] == "mcap" and all(x.endswith(".mcap") for x in
                                                                       meta["relative_file_paths"]),
        meta["storage_identifier"], "rosbag2 storage plugin")
    bad = [t for t, i in topics.items() if (i["type"] != CI if t in IMAGES else i["type"] not in (JS, PS, STR)
           or any(w in t.lower() for w in ("image", "camera", "cam", "video")))]
    if not IMAGES:
        add("no_camera_data", not bad and set(topics) == set(got), dict(bad=bad, topics=sorted(topics)),
            "only JointState, PoseStamped and String topics, none named image/camera/video")
    else:
        add("state_and_camera_topics", not bad and set(topics) == set(got), dict(bad=bad, topics=sorted(topics)),
            "JointState, PoseStamped and String topics plus one CompressedImage topic per camera in capture.json")
        per = {}
        for t, cam in IMAGES.items():
            c = cap["cameras"][cam]
            hz = camera_hz(cap, cam)
            # every frame kept: gz renders at the first 1 ms step at or after k / hz; else capture.py's tolerance
            tol = dt_ns if cap["rate_hz"] >= c["camera_hz"] else round(0.5e9 / c["camera_hz"])
            ns = np.array([b for _, b in stamps[t]], np.int64)
            k = np.round(ns * hz / 1e9).astype(np.int64)
            want_n = int(np.floor(n * dt * hz + 1e-9))
            pngs = frames(replay, n * dt_ns)[cam]
            per[t] = x = dict(
                count=len(ns), want=f"{want_n} +- 1", first_s=ns[0] / 1e9 if len(ns) else None,
                last_s=ns[-1] / 1e9 if len(ns) else None, max_off_grid_ns=int(np.abs(ns - np.round(k * 1e9 / hz)).max())
                if len(ns) else None, missing=int(k[-1] - k[0] + 1 - len(k)) if len(k) else None,
                log_on_tick=all(a == tick_ns(b, hz) for a, b in stamps[t]), in_window=bool(len(ns) and ns[0] >= 0 and ns[-1] < n * dt_ns),
                increasing=bool(np.all(np.diff(k) > 0)), frame_id_format=all(m[:2] == (cam, "jpeg") for m in got[t]),
                jpeg_markers=all(m[2] == b"\xff\xd8\xff\xd9" for m in got[t]),
                pngs_in_window=len(pngs) if pngs else None,
                stamps_eq_pngs=[a for a, _ in pngs] == ns.tolist() if pngs else None)
            x["pass"] = bool(abs(len(ns) - want_n) <= 1 and x["max_off_grid_ns"] is not None and x["max_off_grid_ns"] <= tol
                             and x["missing"] == 0 and x["log_on_tick"] and x["in_window"] and x["increasing"]
                             and x["frame_id_format"] and x["jpeg_markers"] and x["stamps_eq_pngs"] is not False)
        add("image_topics", all(x["pass"] for x in per.values()), per,
            "per camera: floor(sim_s x Hz) +- 1 frames, stamps on the 1 / Hz grid (within 1 ms when every frame is "
            "kept), none missing, log time = the frame's tick k x int(1e9 / Hz) ns, inside the state log, frame_id = camera, format jpeg, "
            "JPEG start and end markers; the PNG stamps when the PNGs are still there")

    want = np.arange(n, dtype=np.int64) * dt_ns
    per = {t: dict(count=len(stamps[t]), log_eq_header=all(a == b for a, b in stamps[t]),
                   stamps_n_ms=np.array_equal(np.array([a for a, _ in stamps[t]], np.int64), want))
           for t in PER_STEP}
    add("one_message_per_step", all(x["count"] == n and x["log_eq_header"] and x["stamps_n_ms"] for x in per.values())
        and cd.get("steps") == str(n), dict(steps=n, topics=per),
        f"per-step topics: count = steps, log time = header stamp = n x {dt_ns} ns for n = 0..steps-1")
    on = {t: dict(count=len(stamps[t]), first_ns=stamps[t][0][0] if stamps[t] else None,
                  on_grid=all(a % dt_ns == 0 and a < n * dt_ns for a, _ in stamps[t]),
                  increasing=all(a[0] < b[0] for a, b in zip(stamps[t], stamps[t][1:])))
          for t in ON_CHANGE}
    add("on_change_stamps", all(x["first_ns"] == 0 and x["on_grid"] and x["increasing"] for x in on.values()),
        on, "on-change topics: first at 0, on the 1 ms grid, strictly increasing")

    diffs = {}
    for t in ("/joint_states", "/action", obj):
        diffs[t] = all(np.array_equal(np.array([m[i] for m in got[t]]).reshape(r_.shape), r_, equal_nan=True)
                       for i, r_ in enumerate(ref[t])) if len(got[t]) == n else False
    for t in ("/base_pose", "/tcp_pose"):
        diffs[t] = len(got[t]) == n and np.array_equal(np.array(got[t]), ref[t])
    for t in ON_CHANGE:   # forward-fill the on-change messages back to one per step
        fill, last, it = [], None, iter(got[t])
        nxt = next(it, None)
        for k in range(n):
            while nxt and nxt[0] == k:
                last, nxt = nxt[1], next(it, None)
            fill.append(last)
        diffs[t] = fill == ref[t]
    add("content_equals_state_log", all(diffs.values()), diffs,
        "every per-step message equals its state_log.npz row; on-change topics forward-filled equal every row")

    fj = list(f["joint_names"])
    names = [f"{robot().model}::{x}" for x in s["robot_joint_names"]] + [str(s["door_joint"])]
    idx = [fj.index(x) for x in names]
    q_last = np.r_[got["/joint_states"][-1][0], got[obj][-1][0]] if n else np.zeros(len(idx))
    q_f, v_f = f["joint_pos"][idx], f["joint_vel"][idx]
    euler = float(np.abs(q_f - (q_last + v_f * dt)).max())
    plain = float(np.abs(q_f - q_last).max())
    add("last_joint_state_is_final_state", int(f["step"]) == n - 1 and euler <= EULER_TOL,
        dict(final_state_step=int(f["step"]), bag_last_step=n - 1, euler_residual=euler, plain_max_diff=plain,
             tol=EULER_TOL),
        "final_state.npz is the state after the last step (step = steps - 1); its positions = the bag's last "
        f"/joint_states + {obj} positions + final velocity x dt within tol (the robot joints + the object joint); "
        "plain_max_diff is the one step of motion")

    failed = json.dumps([x["name"] for x in v["checks"] if not x["pass"]])
    want_cd = dict(plan_run=v["plan_run"], attempt_id=str(v["attempt_id"]), grasp_id=str(v["grasp_id"]),
                   verdict=v["verdict"], failed_checks=failed)
    prompt = episode(replay).get("prompt")
    if prompt:
        want_cd[PROMPT] = prompt
    rec = mcap_metadata(bag / meta["relative_file_paths"][0]).get("rosbag2", {})
    rec_cd = (YAML(typ="safe").load(rec["serialized_metadata"]) or {}).get("custom_data") if rec else None
    ok_cd = (all(cd.get(k) == x for k, x in want_cd.items()) and json.loads(cd.get("verdict_json", "null")) == v
             and set(cd) == set(CUSTOM) | set(want_cd) and rec_cd == cd)
    add("metadata_ids_verdict", ok_cd, dict(keys=sorted(cd), **{k: cd.get(k) for k in want_cd},
                                             mcap_record_equal=rec_cd == cd),
        "metadata.yaml custom_data = verdict.json (ids, verdict, failed checks, whole verdict) and episode.json's "
        f"prompt as {PROMPT}, and the .mcap's rosbag2 metadata record holds the same custom_data")

    side = replay / "bag.json"
    size = sum(p.stat().st_size for p in bag.iterdir())
    rec_size = json.loads(side.read_text()).get("size_bytes") if side.exists() else None
    add("size_recorded", rec_size == size, dict(size_bytes=size, bag_json=rec_size),
        "bag.json beside the bag records its size")

    failed = [x["name"] for x in checks if not x["pass"]]
    print(json.dumps({"bag": str(bag), "replay": str(replay), "pass": not failed, "failed": failed,
                      "checks": checks, "topics": topics}, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    sys.exit({"write": write, "check": check}[cmd](*args))
