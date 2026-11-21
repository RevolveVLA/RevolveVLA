"""The AutoMoMa generator: per config one plan run in the datagen image (cuRobo, GPU), then the planned attempts'
camera-off replay checks in the simulator image, in order, until one passes (N-of-N with `checks`).

  generator: {plugin: automoma, attempts: 50, checks: 3}      # attempts: trajopt problems per plan (generator.yaml),
                                                              # checks: replays that must all pass, variant: datagen variant
Runs on the host (docker SDK; components/datagen/generate.py does the jobs). Recording the accepted attempt with cameras is
the recorder's job, so a split with `record: false` ends here: its start states are the Result.

Result.realised: {"attempt_id", "start_pose" (the accepted attempt's start, object frame), "robot_base.x_m", ".y_m",
".yaw_deg" (offsets from the start-box centre and the facing, comparable with the sampled dims), "grasp_id" and "grasp.id" (when varied),
"grasp.along_m" / ".depth_m" / ".roll_deg" (the grasp by value, when varied)};
a reach / drive start reports "reach_start.<dim>" / "drive_start.<dim>" (grasp_m, d_m, yaw_deg, drive: back_m, side_m,
heading_deg) of the accepted attempt, its jitter included, and its start_pose (the far start of a drive).
Result.paths: plan (the plan run), check (the accepted check dir: actions.npz, live.json, episode.json, state_log.npz),
manifest (generate.py's, per episode dir: also what evaluation/starts.py takes as `--source`)."""
from __future__ import annotations

import json
import math
from pathlib import Path

from datagen.core.errors import DatagenError
from datagen.core.interfaces import Config, Generator, Result
from datagen.core.registry import register

from .. import generate_module
from .layout import Realiser

PARAMS = ("attempts", "checks", "variant")


class AutoMomaError(DatagenError):
    pass


def realised_start(plan_dir: Path, attempt_id: int) -> dict:
    """An attempt's planned start base pose (handoff.npz traj_robot_obj row 0) as the object-frame pose and as offsets
    from the plan's start-box centre / facing (handoff.json starts); {} when the plan has no starts record."""
    import numpy as np
    h = json.loads((plan_dir / "handoff.json").read_text()).get("starts") or {}
    if h.get("mode") in ("reach", "drive"):
        return realised_reach(h, attempt_id)
    if "box_centre_obj" not in h:
        return {}
    z = np.load(plan_dir / "handoff.npz")
    x, y, yaw = (float(v) for v in z["traj_robot_obj"][z["attempt_id"].tolist().index(attempt_id), 0, :3])
    (bx, by), facing = h["box_centre_obj"], h["facing_yaw_obj"]
    return {"start_pose": [x, y, yaw], "robot_base.x_m": x - bx, "robot_base.y_m": y - by,
            "robot_base.yaw_deg": math.degrees((yaw - facing + math.pi) % (2 * math.pi) - math.pi)}


def realised_reach(starts: dict, attempt_id: int) -> dict:
    """realised_start of a reach / drive plan (handoff.json "starts": the sample and each attempt's values)."""
    at = next(a for a in starts["attempts"] if a["attempt_id"] == attempt_id)
    dims = {"d_m": at["d_m"], "yaw_deg": at["dyaw_deg"]}
    dims["grasp_m"] = next(v for k, v in starts["sample"].items() if k.startswith("grasp_"))
    if starts["mode"] == "drive":
        dims.update({k: at[k] for k in ("back_m", "side_m", "heading_deg")})
    return {"start_pose": list(at.get("start_obj", at["base_obj"])),
            **{f"{starts['mode']}_start.{k}": v for k, v in dims.items()}}


@register("generator", "automoma")
class AutoMoma(Generator):
    def __init__(self, spec, params, jobs=None):
        """jobs: generate.Jobs (default: the docker jobs), injected by tests."""
        if unknown := sorted(set(params) - set(PARAMS)):
            raise AutoMomaError(f"generator automoma: unknown params {unknown} (known: {list(PARAMS)})")
        self.spec, self.params, self.jobs, self.realiser = spec, dict(params), jobs, Realiser(spec)
        if not isinstance(self.checks, int) or self.checks < 1:
            raise AutoMomaError(f"generator automoma: checks {self.checks!r} is not an int >= 1")

    @property
    def checks(self):
        return self.params.get("checks", 1)

    def generate(self, config: Config, out: Path) -> Result:
        G = self._generate_module()
        layout = self.realiser.layout(config)
        m = G.generate(self.spec.task, [config.seed], out, 1, 1, self.params.get("attempts"),
                       self.jobs or G.DOCKER, layouts={config.seed: layout} if layout else None, checks=self.checks,
                       record=False, **generate_module.gui_kw(self.spec))
        rec = m["seeds"][str(config.seed)]
        if rec["state"] != "done":
            return Result.fail(self.failure(G, Path(out), rec))
        episode = rec["episodes"][0]
        plan_dir = Path(out) / rec["plan"]["dir"]
        realised = dict(attempt_id=episode["attempt_id"], **realised_start(plan_dir, episode["attempt_id"]),
                        **({"grasp_id": layout["grasp_id"], "grasp.id": layout["grasp_id"]} if "grasp_id" in layout else {}),
                        **{f"grasp.{dim}": v for dim, v in layout.get("grasp", {}).items()})
        return Result(True, "", realised, {"plan": str(plan_dir), "check": str(Path(out) / episode["check"]),
                                           "manifest": str(Path(out) / "manifest.json")})

    def _generate_module(self):
        """generate.py (docker SDK, numpy), imported when first needed; REVOLVE_ROBOT / variant as the CLI sets them."""
        return generate_module.load(self.spec.robot, self.spec.task, self.params.get("variant"))

    @staticmethod
    def failure(G, out: Path, rec) -> str:
        """Why the seed has no accepted check: plan failed / no IK / FK check failed / replay rejected (or the error)."""
        if rec["state"] == "error":
            return rec["reason"]
        p = rec["plan"] or {}
        repeat = [c["repeat_check"]["reason"] for c in rec["checks"].values() if (c.get("repeat_check") or {}).get("reason")]
        return G.plan_failure(p.get("rc"), p.get("planned"), lambda: G.read(out / p["dir"] / "handoff.json"),
                              p.get("ok")) or "replay rejected" + (f" ({repeat[-1]})" if repeat else "")
