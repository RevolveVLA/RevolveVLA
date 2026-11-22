"""The src `fake` rollout plus the state log a state-log judge reads (state_log.npz + live.json in the rollout dir), so c1/c2
are covered end to end on CPU. Registered as rollout plugin `fake_eval_log` (put tests/ on $EVAL_PLUGIN_PATH).

Params as the `fake` rollout (phases, fail_seeds, failed_phase, always_fail, delay_s) and: `push_seeds` (the door is opened
without the handle held: c1 passes, c2 fails), `closed_seeds` (the door stays shut). A rollout whose phases failed leaves the
door shut. Every finished call is appended to $FAKE_EVAL_LOG as `rollout <seed> <dir name>`."""
import json
import os

import numpy as np

from evaluation.core.registry import REGISTRY
from evaluation.plugins.rollouts.fake import FakeRollout

HINGE = np.array([-1.2, 2.4, 1.75])
GRASP = [[0, 0, 0, -1.79], [0, 0, 0, 2.35], [0, 0, 0, 0.97], [0, 0, 0, 1.0]]
DT, N = 0.01, 1400                                      # 14 s: the door opens 0 -> open_deg between 1 s and 11 s


def write_state_log(out, open_deg=65.0, held=True):
    """A rollout dir's log: the door rises to open_deg; the TCP follows the bar (held) or stays 0.5 m away (a push)."""
    q = np.radians(open_deg) * np.clip((np.arange(N) * DT - 1.0) / 10.0, 0, 1)
    d = np.array(GRASP)[:2, 3] - HINGE[:2]
    c, s = np.cos(q), np.sin(q)
    bar = np.stack([HINGE[0] + c * d[0] - s * d[1], HINGE[1] + s * d[0] + c * d[1]], 1)
    pos = np.column_stack([bar if held else bar + 0.5, np.full(N, 1.0)])
    np.savez(out / "state_log.npz", door_pos=q, action_grip=np.zeros(N), tcp_pose=np.column_stack([pos, np.tile([1.0, 0, 0, 0], (N, 1))]),
             handle_in_tcp=HINGE - pos, control_dt=np.array(DT),
             robot_joint_names=np.array(["base_x", "panda_finger_joint_left", "panda_finger_joint_right"]),
             robot_joint_pos=np.column_stack([np.zeros(N), np.full(N, 0.007), np.full(N, 0.007)]),
             contact_left_finger_handle=np.ones(N, bool), contact_right_finger_handle=np.ones(N, bool))
    (out / "live.json").write_text(json.dumps({"grasp_pose_world": GRASP}))


@REGISTRY.register("rollout", "fake_eval_log")
class FakeLogRollout(FakeRollout):
    def __init__(self, spec, params):
        super().__init__(spec, params)
        self.push, self.closed = set(params.get("push_seeds", ())), set(params.get("closed_seeds", ()))

    def run(self, item, out, *, seed, **kw):
        record = super().run(item, out, seed=seed, **kw)
        shut = seed in self.closed or not record["passed"]
        write_state_log(out, open_deg=40.0 if shut else 65.0, held=seed not in self.push)
        if log := os.environ.get("FAKE_EVAL_LOG"):
            with open(log, "a") as f:
                f.write(f"rollout {seed} {out.name}\n")
        return record
