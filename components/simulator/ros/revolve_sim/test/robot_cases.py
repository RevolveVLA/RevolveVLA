"""Per-robot data of the T1 / T2 tests (no ROS imports: used on the host and in the sim container).

CONTROL (check_control.py, test_control.py): the arm and hand JTCs and their joints, the arm targets [rad] (in free
space from the scene start), the hand joint's closed / fully open values and tolerance, and the contact probe: a
static box of side `box` [m] spawned at the TCP (+ `box_offset` in the TCP frame; `box_aligned`: with the TCP's
orientation, else world-aligned) while the hand is open; closing on it must stop the hand joint within `grasp`.
"""
ROBOTS = ["mobile_panda", "mobile_so101"]

CONTROL = {
    "mobile_panda": {
        "arm_controller": "panda_arm_controller",
        "arm_joints": [f"panda_joint{i}" for i in range(1, 8)],
        "arm_targets": {"A": [0.5, -0.3, 0.3, -2.0, 0.3, 1.9, 0.5],
                        "B": [-0.6, 0.2, -0.4, -1.6, -0.3, 2.2, 1.2],
                        "C": [0.0, 0.0, 0.0, -1.2, 0.0, 1.2, 0.785],     # q2 = 0: joints 1 and 3 coaxial
                        "ready": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]},
        "hand_controller": "panda_hand_controller",
        "hand_joint": "panda_finger_joint1",       # [m], per finger
        "closed": 0.0, "open": 0.04, "hand_tol": 0.002,
        "box": 0.03, "box_offset": [0.0, 0.0, 0.0], "box_aligned": False, "grasp": [0.010, 0.020],
    },
    "mobile_so101": {
        "arm_controller": "arm_controller",
        "arm_joints": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        "arm_targets": {"A": [0.6, -0.8, 0.9, 0.6, 0.5],
                        "B": [-0.7, -0.3, 0.6, 0.9, -0.8],
                        "C": [0.0, 0.4, -0.4, 0.8, 1.2],                 # upper arm leaning forward: most gravity load
                        "ready": [0.0, -1.57, 1.57, 0.0, 0.0]},
        "hand_controller": "gripper_controller",
        "hand_joint": "gripper",                   # [rad], the moving jaw
        "closed": -0.174533, "open": 1.74533, "hand_tol": 0.02,
        # a 16 mm cube between the jaw tips, 0.5 mm into the fixed jaw's tip pad (its face 8 mm along TCP +y);
        # the moving jaw stops on it near 0 rad (a 16 mm gap)
        "box": 0.016, "box_offset": [0.0, 0.0005, 0.0], "box_aligned": True, "grasp": [-0.08, 0.08],
    },
}
