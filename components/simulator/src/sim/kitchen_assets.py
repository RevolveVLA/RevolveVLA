"""Kitchen layout of the asset set, chosen by $KITCHEN_ASSETS (partnet, the only set
and the default; the launcher sets it). Everything else about the furniture (joints,
handles, AABBs) comes from the models' manifest.json files.

Model poses must match the world files worlds/kitchen_partnet*.sdf. Values are in
metres / radians; model frames have the front at +x and the floor at z = 0.
"""
import math
import os

PROFILES = {
    "partnet": dict(
        # fridge 11211 against the north wall facing south: hinge on its east side, so the
        # door swings east into the open floor; cabinet 46452 against the west wall facing
        # east, drawer bar at y 0.29.
        poses={"fridge": (-1.5, 2.68, -math.pi / 2), "kitchen_counter": (-2.625, 0.29, 0.0),
               "table_01": (1.5, -0.5, 0.0)},
        door2=None,
        # cavity above the upper wire rack (top 1.1745) up to the ceiling (1.591)
        fridge_inside=((-0.284, 0.258), (-0.267, 0.275), (1.13, 1.59)),
        cup_fridge=(-1.5, 2.56, 1.1765),            # fridge model (0.12, 0, 1.1765)
        # bar is clear of its standoffs between z 0.938 and 1.112 (fingers ~2 cm wide)
        handle_z=1.08,
        handle_band=(0.95, 1.10),
        # handle on the door's west edge: reach it from the west
        handle_bearing=math.radians(-45),
        drawer_bearing=math.radians(45),
        # 10 cm: the joint move home brushed the drawer front (14.8 -> 13.4 cm)
        drawer_release_back=0.20,
        episode=dict(
            v_drive=0.8,                            # base peak speed of the drives [m/s] (90 s budget)
            # door to 95 deg: at 64 its edge leaves < 1 cm for the forearm at the cup
            door_goal=math.radians(95.0), swing_dur=8.0, swing_shift=(0.0, 0.0),
            # the open door blocks the east side: reach the cup from the west;
            # hand x up (link7 above the hand, clear of the rack's front lip)
            cup_yaw=math.radians(100.0), cup_bearing=math.radians(-45), cup_d=0.73,
            cup_delta=math.radians(20.0), cup_roll=1, cup_grasp_z=0.084,
            cup_stage=(-0.15, -0.20),
            back_off=(-0.25, -0.45),
            place_xy=(0.895, -0.30), table_yaw=0.0, table_d=0.70,
            drawer_back=0.3,
        ),
    ),
}

NAME = os.environ.get("KITCHEN_ASSETS", "partnet")
if NAME not in PROFILES:
    raise ValueError(f"KITCHEN_ASSETS={NAME!r}: expected one of {sorted(PROFILES)}")
PROFILE = PROFILES[NAME]
POSES = PROFILE["poses"]
DOOR2 = PROFILE["door2"]
FRIDGE_INSIDE = PROFILE["fridge_inside"]
CUP_FRIDGE = PROFILE["cup_fridge"]
HANDLE_Z = PROFILE["handle_z"]
HANDLE_BAND = PROFILE["handle_band"]
HANDLE_BEARING = PROFILE["handle_bearing"]
DRAWER_BEARING = PROFILE["drawer_bearing"]
DRAWER_RELEASE_BACK = PROFILE["drawer_release_back"]
EPISODE = PROFILE["episode"]
