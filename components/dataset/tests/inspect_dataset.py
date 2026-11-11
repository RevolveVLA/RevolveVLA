"""Test helper run INSIDE revolve-dataset (conftest.in_image): prints one JSON line.

  inspect_dataset.py values <dataset dir> <episode dir>...   per episode, in order: LeRobotDataset load,
      frames decoded (first / middle / last: image shape, dtype, range), observation.state and action
      against the episode's state_log.npz: each frame's (state, action) must be one logged 1 ms step (the
      contract's `hold` rule), the steps one 1/30 s tick apart (+-1 ms); the meta/episodes labels.
  inspect_dataset.py compare <dataset a> <dataset b> [--strict]   data parquet values per episode (all
      columns but the global index / episode_index unless --strict), episode lengths, tasks; --strict also
      compares meta/info.json, meta/stats.json, the episodes metadata LeRobot writes (common columns) and
      the decoded video frames (and reports whether the mp4 bytes are identical).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/repo/components/dataset/src/dataset")
sys.path.insert(0, "/repo/common")
import labels as L  # noqa: E402
from revolve_common import policy_io  # noqa: E402

REPO_ID = "revolve/dataset"


def data(root):
    return pd.concat([pd.read_parquet(p) for p in sorted((Path(root) / "data").rglob("*.parquet"))], ignore_index=True)


def episodes(root):
    return pd.concat([pd.read_parquet(p) for p in sorted((Path(root) / "meta" / "episodes").rglob("*.parquet"))],
                     ignore_index=True).sort_values("episode_index").reset_index(drop=True)


def values(root, ep_dirs):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(REPO_ID, root=root)
    fps = ds.meta.fps
    eps = episodes(root)
    df = data(root)
    out = []
    for i, ep in enumerate(ep_dirs):
        row = eps[eps.episode_index == i].iloc[0]
        rows = df[df.episode_index == i]
        z = np.load(Path(ep) / "state_log.npz")
        state = np.stack(rows["observation.state"].to_numpy())
        action = np.stack(rows["action"].to_numpy())
        # every step in the policy format (policy_io): measured twist + arm/gripper joints, commanded twist + targets
        dt = float(z["control_dt"])
        base = z["robot_joint_pos"][:, :3]
        log = np.column_stack([policy_io.measured_twist(base, fps, dt), z["robot_joint_pos"][:, 3:],
                               policy_io.command_twist(z["action_base"], fps, dt), z["action_arm"], z["action_grip"]])
        got = np.column_stack([state, action])
        err = np.array([np.abs(log - g).max(axis=1) for g in got])        # frames x steps
        step = err.argmin(axis=1)
        frame_err = err[np.arange(len(got)), step]
        dt_ms = 1000.0 / fps
        span_ok = abs((step[-1] - step[0]) - (len(got) - 1) * dt_ms) <= 1.0
        frames = {}
        for name, j in (("first", 0), ("middle", len(rows) // 2), ("last", len(rows) - 1)):
            item = ds[int(row.dataset_from_index) + j]
            frames[name] = {key: dict(shape=list(item[key].shape), dtype=str(item[key].dtype),
                                      min=float(item[key].min()), max=float(item[key].max()))
                            for key in ds.meta.camera_keys}
            frames[name]["task"] = item["task"]
        out.append(dict(length=len(rows), meta_length=int(row.length), state_action_max_abs_err=float(frame_err.max()),
                        first_step=int(step[0]), last_step=int(step[-1]), step_span_ok=bool(span_ok), frames=frames,
                        labels={k: getattr(v, "item", lambda: v)() for k, v in L.of_row(row).items()}, state_dtype=str(state.dtype), action_dtype=str(action.dtype)))
    return dict(episodes=out, total_frames=len(ds), camera_keys=list(ds.meta.camera_keys),
                features=json.loads((Path(root) / "meta" / "info.json").read_text())["features"])


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def decoded(root, key):
    import av
    frames = []
    for p in sorted((Path(root) / "videos" / key).rglob("*.mp4")):
        with av.open(str(p)) as c:
            frames += [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    return np.stack(frames)


def compare(a, b, strict=False):
    da, db = data(a), data(b)
    ea, eb = episodes(a), episodes(b)
    diffs = []
    if list(ea.length) != list(eb.length):
        diffs.append(f"lengths {list(ea.length)} != {list(eb.length)}")
    skip = set() if strict else {"index", "episode_index"}
    cols = sorted(set(da.columns) | set(db.columns))
    for c in cols:
        if c in skip:
            continue
        if c not in da or c not in db:
            diffs.append(f"data column {c} only in {'a' if c in da else 'b'}")
            continue
        va, vb = np.stack(da[c].to_numpy()), np.stack(db[c].to_numpy())
        if va.shape != vb.shape or va.tobytes() != vb.tobytes():
            diffs.append(f"data column {c} differs")
    ta, tb = (pd.read_parquet(Path(r) / "meta" / "tasks.parquet").index.tolist() for r in (a, b))
    if ta != tb:
        diffs.append(f"tasks {ta} != {tb}")
    res = dict(frames=[len(da), len(db)], episodes=[len(ea), len(eb)], data_columns=len(cols))
    if strict:
        ia, ib = (json.loads((Path(r) / "meta" / "info.json").read_text()) for r in (a, b))
        if ia != ib:
            diffs.append("meta/info.json differs: " + ", ".join(k for k in set(ia) | set(ib) if ia.get(k) != ib.get(k)))
        sa, sb = (json.loads((Path(r) / "meta" / "stats.json").read_text()) for r in (a, b))
        if sa != sb:
            diffs.append("meta/stats.json differs: " + ", ".join(sorted(k for k in set(sa) | set(sb) if sa.get(k) != sb.get(k))))
        common = sorted(set(ea.columns) & set(eb.columns))
        res["episode_columns_only_in"] = dict(a=sorted(set(ea.columns) - set(eb.columns)),
                                              b=sorted(set(eb.columns) - set(ea.columns)))
        for c in common:
            if ea[c].astype(str).tolist() != eb[c].astype(str).tolist():
                diffs.append(f"meta/episodes column {c} differs")
        res["videos"] = {}
        for key in sorted(p.name for p in (Path(a) / "videos").iterdir()):
            fa, fb = decoded(a, key), decoded(b, key)
            same_frames = fa.shape == fb.shape and fa.tobytes() == fb.tobytes()
            files = sorted(p.relative_to(a) for p in (Path(a) / "videos" / key).rglob("*.mp4"))
            res["videos"][key] = dict(frames=[len(fa), len(fb)], decoded_identical=same_frames,
                                      mp4_bytes_identical=all(sha(Path(a) / f) == sha(Path(b) / f) for f in files))
            if not same_frames:
                diffs.append(f"video {key}: decoded frames differ")
    return dict(res, diffs=diffs, identical=not diffs)


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    if cmd == "values":
        r = values(args[0], args[1:])
    else:
        r = compare(args[0], args[1], strict="--strict" in args)
    print(json.dumps(r, default=str))
