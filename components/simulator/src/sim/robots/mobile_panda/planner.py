"""cuRobo planner robot of the mobile Panda (datagen: gen_assets.robot() adds the shared base chain, the base
spheres and the kinematics cfg; robots/base.py Robot.planner loads this module). Runs in revolve-datagen only.

  chain(robot, rd)               the arm from panda_link0 to the TCP: franka_description robots/fer kinematics
                                 and limits, the hand at HAND_YAW, the TCP tcp.hand_offset_z along the hand z,
                                 both fingers (as robots/mobile_panda/model.py's finger links); cuRobo's
                                 franka.yml spheres per link
  self_collision(robot, rd, links)   franka.yml's self-collision ignore pairs and buffers
  BOX_CENTRE_OBJ                 start design (start_design.py): box centre per task in the object frame
"""
import math

import yaml

import gen_assets as ga
import specs_bridge as plugins
from robots.mobile_panda.robot import HAND_YAW

# Box centre [m] per task in the object frame, in front of the handle (a starts block needs one). A D1 plan
# smoke (seed 60, 50 attempts) found no start IK for drawer bases under x 0.85 m (the counter) and few fridge
# trajectories from y under -0.28 m (the handle side of the door).
BOX_CENTRE_OBJ = {"open_fridge_door": (0.95, -0.05), "open_drawer": (0.98, 0.0)}

ARM_LINKS = [f"panda_link{n}" for n in range(8)] + ["panda_hand", "panda_leftfinger", "panda_rightfinger"]
DROP = "attached_object"


def _franka():
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml
    return load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]["kinematics"]


def chain(robot, rd):
    """dict(links: link names root..TCP, joints: URDF joints, spheres: {link: [{center, radius}]})."""
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml
    fd = plugins.path(rd["description"]) / "robots/fer"
    kin = yaml.safe_load((fd / "kinematics.yaml").read_text())
    lim = yaml.safe_load((fd / "joint_limits.yaml").read_text())
    joints = []
    for n in range(1, 8):
        k = kin[f"joint{n}"]["kinematic"]
        lj = lim[f"joint{n}"].get("limit", lim[f"joint{n}"])
        joints.append(ga.joint(f"panda_joint{n}", "revolute", f"panda_link{n - 1}", f"panda_link{n}",
                               ga.sdf.tf((k["x"], k["y"], k["z"]), (k["roll"], k["pitch"], k["yaw"])),
                               (0, 0, 1), (lj["lower"], lj["upper"], lj["velocity"])))
    joints += [
        ga.joint("panda_hand_joint", "fixed", "panda_link7", "panda_hand",
                 ga.sdf.tf((0, 0, kin["joint8"]["kinematic"]["z"]), (0, 0, HAND_YAW))),
        ga.joint("panda_tcp_joint", "fixed", "panda_hand", rd["tcp"]["link"], ga.sdf.tf((0, 0, rd["tcp"]["hand_offset_z"])))]
    for side, fyaw in (("left", 0.0), ("right", math.pi)):      # as model.py's finger links
        joints.append(ga.joint(f"panda_finger_joint_{side}", "prismatic", "panda_hand",
                               f"panda_{side}finger", ga.sdf.tf((0, 0, 0.0584), (0, 0, fyaw)), (0, 1, 0),
                               (rd["gripper"]["closed_m"], rd["gripper"]["open_m"], 0.2)))
    spheres = load_yaml(join_path(get_robot_configs_path(), _franka()["collision_spheres"]))["collision_spheres"]
    spheres = {n: spheres[n] for n in ARM_LINKS}
    for s in spheres["panda_rightfinger"]:      # gz's right finger frame is yawed by pi
        s["center"] = [-s["center"][0], -s["center"][1], s["center"][2]]
    return dict(links=[*ARM_LINKS, rd["tcp"]["link"]], joints=joints, spheres=spheres)


def self_collision(robot, rd, links):
    """(ignore, buffer) over the arm's sphere links (cuRobo franka.yml, without its attached object)."""
    franka = _franka()
    ignore = {k: [x for x in v if x != DROP] for k, v in franka["self_collision_ignore"].items()}
    buffer = {k: v for k, v in franka["self_collision_buffer"].items() if k != DROP}
    return ignore, buffer
