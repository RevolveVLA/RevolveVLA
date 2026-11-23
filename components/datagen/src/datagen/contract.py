"""The datagen output against DATAGEN-v2.md: `manifest.json` (8.4: schema, every path exists, `passed` as in the result file,
the scene instances validate) and the episode bags (7.2 frozen topics, 7.4 the 30 Hz action series, JPEG cameras at ~30 Hz,
`/tf_static`). Pure python except `read_bag` (rosbag2_py: the simulator image), so `check_manifest` and `check_bag` run
on the host and the bag part runs in a client container:

    python3 -m datagen.contract --out <datagen out> --record images|no-images     (prints one RESULT json line, rc 1 on a problem)

`action_series` is the one of components/simulator/ros/revolve_sim/test/check_contract.py (7.4), ported."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ACTION_HZ = 30.0
CMD_VEL_TIMEOUT_S = 0.5                     # the omni controller's cmd_vel_timeout: no command for longer = zero twist
CAMERA_HZ = (25.0, 35.0)                    # "~30 Hz"
ON_DEMAND = ("/sensor/contact/", "/joint_trajectory", "/cmd_vel")       # messages only while in contact / commanded
KEEP_IMAGES = 5                             # messages kept per camera topic (the format check)
EPISODE_KEYS = {"seed", "scene", "robot", "task", "episode", "result", "bag", "passed", "datagen"}
REPO = Path(__file__).resolve().parents[4]


def stamp(m) -> float:
    return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9


def action_series(states: dict, cmds: list, t0: float, t1: float):
    """The 30 Hz action on [t0, t1]: per JTC its controller_state reference positions (the last state at or before each
    tick), the base twist (the last cmd_vel within CMD_VEL_TIMEOUT_S, else zero) -> (grid, {name: array})."""
    grid = np.arange(t0, t1, 1.0 / ACTION_HZ)
    out = {}
    for name, ms in states.items():
        st = np.array([stamp(m) for m in ms])
        ref = np.array([m.reference.positions for m in ms])
        out[name] = ref[np.clip(np.searchsorted(st, grid, side="right") - 1, 0, len(st) - 1)]
    ct = np.array([stamp(m) for m in cmds]) if cmds else np.zeros(0)
    cv = np.array([[m.twist.linear.x, m.twist.linear.y, m.twist.angular.z] for m in cmds]) if cmds else np.zeros((0, 3))
    base = np.zeros((len(grid), 3))
    if len(ct):
        i = np.searchsorted(ct, grid, side="right") - 1
        ok = (i >= 0) & (grid - ct[np.clip(i, 0, None)] <= CMD_VEL_TIMEOUT_S)
        base[ok] = cv[i[ok]]
    out["cmd_vel"] = base
    return grid, out


def check_manifest(out: Path, instances: bool = True) -> list[str]:
    """Problems of `<out>/manifest.json` (empty = valid): the 8.4 schema, counts, every path exists, `passed` is the result
    file's, a bag path is a directory; instances (`instances`): every scene.yaml loads as a scene instance."""
    out, bad = Path(out), []
    try:
        m = json.loads((out / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as e:
        return [f"manifest.json: {e}"]
    bad += [f"manifest.{k} missing" for k in ("task", "robot", "created", "episodes", "counts") if k not in m]
    eps = m.get("episodes", [])
    for e in eps:
        tag = f"episode (seed {e.get('seed')}, n {e.get('episode')})"
        bad += [f"{tag}: {k} missing" for k in sorted(EPISODE_KEYS - set(e))]
        if EPISODE_KEYS - set(e):
            continue
        bad += [f"{tag}: {k} {e[k]!r} != {m[k]!r}" for k in ("task", "robot") if e[k] != m.get(k)]
        bad += [f"{tag}: datagen.{k} missing" for k in ("expert", "attempt", "wall_s") if k not in e["datagen"]]
        for k in ("scene", "result"):
            if not (out / e[k]).is_file():
                bad.append(f"{tag}: {k} {e[k]} does not exist")
        if e["bag"] is not None and not (out / e["bag"] / "metadata.yaml").is_file():
            bad.append(f"{tag}: bag {e['bag']} is not a bag directory")
        if (out / e["result"]).is_file() and json.loads((out / e["result"]).read_text()).get("passed") != e["passed"]:
            bad.append(f"{tag}: passed differs from {e['result']}")
    counts = m.get("counts", {})
    want = {"scenes": len({e.get("seed") for e in eps}), "attempted": len(eps), "passed": sum(bool(e.get("passed")) for e in eps)}
    bad += [f"counts.{k} {counts.get(k)} != {v}" for k, v in want.items() if counts.get(k) != v]
    if instances:
        from revolve_common import specs
        for scene in sorted({e["scene"] for e in eps if "scene" in e and (out / e["scene"]).is_file()}):
            try:
                specs.load(out / scene, assets=False)
            except specs.SpecError as err:
                bad.append(f"instance {scene}: {err}")
    return bad


def check_bag(types: dict, times: dict, msgs: dict, expected: list[str], images: bool) -> tuple[list[str], dict]:
    """Problems of one episode bag + a report. `types` {topic: type}, `times` {topic: [stamp s]} (every topic), `msgs`
    {topic: [message]} (controller_state, cmd_vel; the first KEEP_IMAGES of each camera); `expected` = the frozen topics
    of the scene and robot. Every topic must be in the bag with messages, except the on-demand ones (absent = none sent);
    `/tf_static` always; cameras (`images`) JPEG at ~30 Hz; a 30 Hz action series extractable, finite, at 30 Hz."""
    on_demand = lambda t: any(k in t for k in ON_DEMAND)                                    # noqa: E731
    bad = [f"topic {t} missing" for t in expected if t not in types and not on_demand(t)]
    bad += [f"topic {t} has no messages" for t in expected if t in types and not times[t] and not on_demand(t)]
    if "/tf_static" not in types:
        bad.append("topic /tf_static missing")
    report = {"topics": len(types), "extra": sorted(set(types) - set(expected))}
    cams = [t for t in expected if t.endswith("/image/compressed")] if images else []
    rates = {}
    for t in cams:
        ts = times.get(t, [])
        rates[t] = round((len(ts) - 1) / (ts[-1] - ts[0]), 2) if len(ts) > 1 and ts[-1] > ts[0] else 0.0
        if not CAMERA_HZ[0] <= rates[t] <= CAMERA_HZ[1]:
            bad.append(f"camera {t} runs at {rates[t]} Hz")
        if not msgs.get(t) or not all("jpeg" in m.format for m in msgs[t]):
            bad.append(f"camera {t} is not JPEG")
    report["camera_hz"] = rates
    jtcs = [t for t in expected if t.endswith("/controller_state")]
    cmd = next((t for t in expected if t.endswith("/cmd_vel")), None)
    states = {t.split("/")[1]: msgs.get(t, []) for t in jtcs}
    if not jtcs or not all(states.values()):
        return bad + ["no controller_state to extract the action from"], report
    t0, t1 = max(stamp(ms[0]) for ms in states.values()), min(stamp(ms[-1]) for ms in states.values())
    if t1 - t0 <= 1.0 / ACTION_HZ:
        return bad + [f"controller_state spans {t1 - t0:.3f} s: no action series"], report
    grid, series = action_series(states, msgs.get(cmd, []) if cmd else [], t0, t1)
    hz = (len(grid) - 1) / (grid[-1] - grid[0])
    report["action"] = {"n": len(grid), "duration_s": round(t1 - t0, 3), "hz": round(hz, 3),
                        "dims": {k: list(v.shape) for k, v in series.items()}}
    if abs(hz - ACTION_HZ) > 0.5:
        bad.append(f"action series at {hz:.2f} Hz")
    if not all(np.isfinite(v).all() for v in series.values()):
        bad.append("action series has non-finite values")
    return bad, report


def read_bag(path, decode: list[str]):
    """-> ({topic: type}, {topic: [stamp s]}, {topic: [message]} for `decode`; cameras keep the first KEEP_IMAGES). Needs
    rosbag2_py (the simulator image)."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"), rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    times, msgs = {t: [] for t in types}, {t: [] for t in decode}
    while r.has_next():
        topic, data, t = r.read_next()
        times[topic].append(t * 1e-9)
        if topic in msgs and not (topic.endswith("/image/compressed") and len(msgs[topic]) >= KEEP_IMAGES):
            msgs[topic].append(deserialize_message(data, get_message(types[topic])))
    return types, times, msgs


def frozen_topics(repo: Path, scene: Path, robot: str, images: bool) -> list[str]:
    """config/record.yaml filled in for the scene instance and robot (the simulator's own function, in the simulator image)."""
    sys.path.append(str(Path(repo) / "components/simulator/ros/revolve_sim/test"))
    from check_contract import frozen_topics as frozen
    return frozen(repo, str(scene), robot, images)


def check_bags(out: Path, record: str, repo: Path = REPO) -> tuple[list[str], dict]:
    """check_bag on every bag of `<out>/manifest.json` -> (problems, {bag: report})."""
    out, bad, reports = Path(out), [], {}
    manifest = json.loads((out / "manifest.json").read_text())
    for e in manifest["episodes"]:
        if e["bag"] is None:
            continue
        expected = frozen_topics(repo, (out / e["scene"]).resolve(), manifest["robot"], record == "images")
        decode = [t for t in expected if t.endswith(("/controller_state", "/cmd_vel", "/image/compressed"))]
        problems, reports[e["bag"]] = check_bag(*read_bag(out / e["bag"], decode), expected, record == "images")
        bad += [f"{e['bag']}: {p}" for p in problems]
    return bad, reports


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="datagen.contract", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--record", choices=("images", "no-images"), default="images")
    ap.add_argument("--repo", default=str(REPO))
    a = ap.parse_args(argv)
    problems, reports = check_bags(Path(a.out), a.record, Path(a.repo))
    print("RESULT " + json.dumps({"problems": problems, "bags": reports}), flush=True)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
