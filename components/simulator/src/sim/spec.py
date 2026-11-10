"""The specs (specs/tasks/<t>/task.yaml for the robot $REVOLVE_ROBOT -> robot + scene, loaded and validated
by revolve_common) as the ported sim code reads them. Replaces the earlier pipeline's plugin loader and per-task validators.

Every threshold comes from the task spec: the live phase gates are its phase conditions (evaluated by
revolve_common.metrics in phase_replay_system), and the offline verdict's legacy keys (episode.json
"thresholds") are filled from the same conditions' `offline:` values here (legacy_thresholds).
Paths in specs are relative to the repo root: $REPO (in the containers /repo).
"""
import math
import os
from functools import lru_cache
from pathlib import Path

from revolve_common import specs

ROOT = Path(os.environ.get("REPO", Path(__file__).resolve().parents[4]))
SPECS = Path(os.environ.get("REVOLVE_SPECS", ROOT / "specs"))
HERE = Path(__file__).resolve().parent              # the sim modules (GZ_SIM_SYSTEM_PLUGIN_PATH)
WORLDS = HERE.parent / "worlds"
# Simulator settings, not task thresholds (they stay with the simulator)
SIM = dict(control_dt=0.001,       # one whole-body Action per physics step [s] (robots.base.CONTROL_DT)
           real_time_factor=0,     # replays uncapped: bit-identical state logs to capped (plugin.yaml note)
           contact_rate_hz=200,    # robot contact sensors (robots/<robot>/model.py); the log holds each message this long
           module="automoma_replay_system")
# Plans and episodes made before the 2026-11-03 renames name the old plugins
RENAMED = {"fridge_door": "open_fridge_door", "drawer_open": "open_drawer", "kitchen": "kitchen_partnet"}


def path(p):
    return ROOT / p


def model_dir(dirs, uri):
    """<dir>/<name>/ for model://<name>: the first of dirs (repo-relative) that has it, as
    GZ_SIM_RESOURCE_PATH resolves it."""
    name = uri.replace("model://", "")
    for d in dirs:
        if (path(d) / name / "model.sdf").is_file():
            return path(d) / name
    raise FileNotFoundError(f"{uri} not in {dirs}")


@lru_cache(maxsize=None)
def task(name) -> specs.Resolved:
    """The task spec resolved for the robot $REVOLVE_ROBOT and its scene (validated)."""
    name = RENAMED.get(name, name)
    return specs.load(SPECS / "tasks" / name / "task.yaml")


def phase(r, name):
    return r.task.phase(name)


def cond(r, phase_name, metric):
    """The first condition on `metric` in a phase (None if the phase has none)."""
    return next((c for c in r.task.phase(phase_name).pass_.all if c.metric == metric), None)


def sanity(r, metric):
    return next(c for c in r.task.sanity.timed + r.task.sanity.untimed if c.metric == metric)


def open_cond(r):
    """The open phase's joint condition (joint_deg / joint_cm, with hold_s)."""
    return next(c for c in r.task.phase("open").pass_.all if c.metric in ("joint_deg", "joint_cm"))


def hold(r):
    """validator.py HOLD: (grip part, part held open, unit, unit -> SI, SI -> unit)."""
    h = r.task.object.hold
    conv = (math.radians, math.degrees) if h.unit == "deg" else (lambda v: v / 100, lambda v: v * 100)
    return (h.grip_part, h.part, h.unit, *conv)


def grasp_half_width(r, size):
    """Per-finger opening the gz grasp check expects: half the smallest handle size over the task's axes."""
    return min(size[a] for a in r.task.object.grasp_half_width_axes) / 2


def offline(c):
    return c.offline if c.offline is not None else c.threshold


def legacy_thresholds(r):
    """episode.json "thresholds" (the offline verdict's keys, validation.py) from the task spec."""
    _, part, unit, _, _ = hold(r)
    oc = open_cond(r)
    return {f"{part}_open_min_{unit}": offline(oc), "hold_s": oc.hold_s,
            "handle_slip_max_m": offline(cond(r, "open", "handle_slip_m")),
            "spawn_clearance_min_m": r.task.replay.spawn_clearance_min_m,
            "peak_body_speed_max_mps": offline(sanity(r, "peak_body_speed_mps")),
            "penetration_max_m": offline(sanity(r, "penetration_m")),
            "grasp_width_tol_m": offline(cond(r, "grasp", "grasp_width_error_m"))}


def legacy_task(r):
    """The old task plugin dict the ported replay.py builds an episode from."""
    t = r.task
    rp = t.replay.model_dump()
    th = legacy_thresholds(r)
    _, part, unit, _, _ = hold(r)
    rp.update({f"{part}_open_min_{unit}": th[f"{part}_open_min_{unit}"], "hold_s": th["hold_s"],
               "handle_slip_max_m": th["handle_slip_max_m"], "budget_s": t.budget_s})
    return dict(name=t.name, object=t.object.name, joint=t.object.joint, handle=t.object.handle,
                joint_topic=t.object.joint_topic, gz_model=t.object.gz_model, prompt=t.prompt,
                world_pose=t.object.world_pose, replay=rp)


def legacy_scene(r):
    s = r.scene
    return dict(name=s.name, world=s.world, assets=s.assets, skip=s.skip, detailed=s.detailed)


def resource_path(r):
    """GZ_SIM_RESOURCE_PATH: the scene's asset dirs, in order."""
    return ":".join(str(path(d)) for d in r.scene.assets)
