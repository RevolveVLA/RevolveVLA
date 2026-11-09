"""Progress parser for revolve_common.wrapper (--parser progress:parse): the PROGRESS {json} lines of
entry.py and generate.py, plus the planner's own output (plan.py per-grasp IK lines, its report and
fk_check.py's summary JSON, both printed with indent=1)."""
import re

from revolve_common.wrapper import json_progress

GRASP = re.compile(r"^grasp (\d+): ")
KEY = re.compile(r'^ "(planned|attempts|trajopt_success|passed|max_pos_err_m)": ([0-9.eE+-]+|true|false),?$')
NAMES = {"planned": "planned", "attempts": "attempts", "trajopt_success": "trajopt_success",
         "passed": "fk_passed", "max_pos_err_m": "fk_max_pos_err_m"}


def parse(line):
    d = json_progress(line)
    if d is not None:
        return d
    m = GRASP.match(line)
    if m:
        return {"stage": "ik_traj", "grasps_done": int(m.group(1)) + 1}
    m = KEY.match(line)
    if m:
        v = m.group(2)
        return {NAMES[m.group(1)]: v == "true" if v in ("true", "false") else float(v) if "." in v or "e" in v else int(v)}
    return None
