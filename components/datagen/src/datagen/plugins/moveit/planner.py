"""MoveIt 2 planning for the expert (moveit_py, imported inside the client container only): OMPL to a joint goal, Pilz
LIN / CIRC to a TCP pose, the planning scene from the scene's collision boxes in the base frame, execution through the
robot's MoveIt controllers (the arm's FollowJointTrajectory). MoveIt plans in `base_footprint`; the TCP poses it takes are
in that frame."""
from __future__ import annotations

import numpy as np

from .geometry import quat


class ArmPlanner:
    def __init__(self, repo, robot: str, urdf: str, group: str, tcp: str, plan_frame: str, node_name: str):
        """`urdf`: the sim's /robot_description (never re-run xacro: DATAGEN 6)."""
        from moveit.planning import MoveItPy
        from revolve_sim.moveit import moveit_params
        self.mp = MoveItPy(node_name=node_name, config_dict=moveit_params(repo, robot, urdf))
        self.group, self.tcp, self.plan_frame = group, tcp, plan_frame
        self.arm = self.mp.get_planning_component(group)
        self.psm = self.mp.get_planning_scene_monitor()
        self.model = self.mp.get_robot_model()

    @staticmethod
    def pose_msg(T):
        from geometry_msgs.msg import Pose
        p = Pose()
        p.position.x, p.position.y, p.position.z = (float(v) for v in T[:3, 3])
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat(T[:3, :3])
        return p

    def scene_boxes(self, boxes, T_base, skip_link=None) -> None:
        """The planning scene = the collision `boxes` (world frames) seen from the base at T_base; `skip_link`'s are left
        out (the bar goes between the fingers for the final LIN)."""
        from moveit_msgs.msg import CollisionObject
        from shape_msgs.msg import SolidPrimitive
        T_inv = np.linalg.inv(T_base)
        with self.psm.read_write() as sc:
            sc.remove_all_collision_objects()
            for b in boxes:
                if b.link == skip_link:
                    continue
                o = CollisionObject()
                o.header.frame_id, o.id, o.operation = self.plan_frame, f"{b.link}::{b.name}", CollisionObject.ADD
                o.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=b.size)]
                o.primitive_poses = [self.pose_msg(T_inv @ b.T)]
                sc.apply_collision_object(o)
            sc.current_state.update()

    def ik(self, seed, T) -> tuple[bool, object]:
        """IK of the TCP pose T (base frame) from the joint values `seed` -> (found, the RobotState)."""
        from moveit.core.robot_state import RobotState
        g = RobotState(self.model)
        g.set_to_default_values()
        g.set_joint_group_positions(self.group, np.array(seed))
        found = g.set_from_ik(self.group, self.pose_msg(T), self.tcp, 0.2)
        g.update()
        return bool(found), g

    def run(self, planner: str, goal=None, pose=None, constraints=None, scale=None) -> bool:
        """Plan with the plan-request parameters `planner` (ompl_rrtc, pilz_lin, pilz_circ) to a RobotState `goal` or a TCP
        `pose` (base frame), then execute -> success."""
        from geometry_msgs.msg import PoseStamped
        from moveit.planning import PlanRequestParameters
        from moveit_msgs.msg import Constraints
        self.arm.set_start_state_to_current_state()
        if goal is not None:
            self.arm.set_goal_state(robot_state=goal)
        else:
            ps = PoseStamped()
            ps.header.frame_id, ps.pose = self.plan_frame, self.pose_msg(pose)
            self.arm.set_goal_state(pose_stamped_msg=ps, pose_link=self.tcp)
        if constraints is not None:
            self.arm.set_path_constraints(constraints)
        prp = PlanRequestParameters(self.mp, planner)
        if scale is not None:
            prp.max_velocity_scaling_factor = prp.max_acceleration_scaling_factor = scale
        plan = self.arm.plan(single_plan_parameters=prp)
        if constraints is not None:
            self.arm.set_path_constraints(Constraints())
        return bool(plan) and bool(self.mp.execute(plan.trajectory, controllers=[]))

    def circ_constraint(self, center):
        """Pilz CIRC path constraint `center` (a point in the base frame): start and goal must be equidistant from it."""
        from geometry_msgs.msg import Pose
        from moveit_msgs.msg import Constraints, PositionConstraint
        c = Constraints(name="center")
        pc = PositionConstraint()
        pc.header.frame_id, pc.link_name = self.plan_frame, self.tcp
        cp = Pose()
        cp.position.x, cp.position.y, cp.position.z = (float(v) for v in center)
        pc.constraint_region.primitive_poses = [cp]
        c.position_constraints = [pc]
        return c
