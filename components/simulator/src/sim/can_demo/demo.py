"""The Panda-only can-on-plate demo (`cli.py can-demo`): a fixed-base Panda + Hand on a table picks a can
lying on its side and places it on a plate (can_on_plate.sdf, controller pick_place_system.py, plan
panda_plan.py, driver pick_place.py). Kept apart from the robot-generic simulator: it is not a task of
the pipeline and runs no robot spec. Its model ("panda", runs/models/plain/panda) is the mobile Panda's
generator without the base (robots/mobile_panda/model.py, mobile=False)."""
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROBOT = "mobile_panda"           # the robot whose arm, laws and model generator the demo uses


def run(a, cli):
    """cli: the simulator's cli module (model cache, world camera regexes, gz runner)."""
    import robots
    import spec
    from robots.mobile_panda import model
    out = cli.rpath(a.out)
    out.mkdir(parents=True, exist_ok=True)
    panda = robots.load(ROBOT)
    plain = spec.path(panda.spec.camera_model).parent.parent / "plain"
    cli._generate(plain / "panda", lambda d: model.main(panda, spec.path(panda.spec.description), d, mobile=False))
    w = (HERE / "can_on_plate.sdf").read_text()
    if not a.video:
        w = cli.SENSORS_RE.sub("", cli.CAMERA_RE.sub("", w))
    (out / "world.sdf").write_text(w)
    cli.progress(stage="gz", what="can-demo")
    rc = cli.run_gz(out, str(plain), [sys.executable, str(HERE / "pick_place.py"),
                                      *(["--video", str(out / "video.mp4")] if a.video else [])], render=a.video,
                    env=dict(GZ_SIM_SYSTEM_PLUGIN_PATH=f"{HERE}:{cli.SIM}:{os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', '')}"))
    log = (out / "driver.log").read_text()
    m = re.search(r"DONE (SUCCESS|FAILED).*", log)
    d = re.search(r"([0-9.]+) cm from", m.group(0)) if m else None
    res = dict(label="CAN_DEMO", pass_=rc == 0, done=m.group(0) if m else None,
               distance_cm=float(d.group(1)) if d else None)
    (out / "result_sim.json").write_text(json.dumps(res, indent=1))
    print(f"CAN_DEMO {'PASS' if rc == 0 else 'FAIL'}: {res['done']}")
    return rc
