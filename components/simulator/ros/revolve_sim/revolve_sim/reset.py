"""The simulator's reset service, for clients running episodes back to back on one sim (datagen, recovery,
evaluation): /simulator/reset (std_srvs/Trigger) puts the sim back to the scene's initial state, through upstream
interfaces only:
  1. controller_manager switch_controller: deactivate the active controllers;
  2. the gz world reset (ros_gz_bridge service /world/<w>/control, ros_gz_interfaces/srv/ControlWorld, reset all:
     the request ROS 2 Simulation Interfaces ResetSimulation sends): every model, the robot included (it is part of
     the world), back to its initial pose and joint positions, sim time back to its start (1 s);
  3. once /clock has jumped back: activate the controllers again, so they latch onto the reset state (the
     gz_ros2_control reset patch keeps the controller manager across the reset); an omni base controller gets one
     zero twist (its no-command placeholder is stamped before the reset, so it would warn until sim time gets there);
  4. every joint_trajectory_controller drives its joints to their ros2_control initial values (robot_start.arm, the
     gripper's start: the robot description's position initial_value) over MOVE_S, until they stay within
     1 deg / 1 mm for HOLD_S (the restarted PID integrators take over gravity again);
  5. with a task: wait until the evaluator has seen the reset (its /evaluator/state `resets` count goes up; it ends
     a running episode with reason `reset` and is not ready until its inputs are live again).
The client then waits for an /evaluator/state with `resets` above its value before the call and "ready": true, and
starts the next episode with /evaluator/start.

  ros2 run revolve_sim reset --ros-args -p world:=<world> -p use_sim_time:=true
"""
import json
import math
import threading
import time
import xml.etree.ElementTree as ET

import rclpy
from controller_manager_msgs.srv import ListControllers, SwitchController
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from ros_gz_interfaces.srv import ControlWorld
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

JTC = 'joint_trajectory_controller/JointTrajectoryController'
OMNI = 'omni_wheel_drive_controller/OmniWheelDriveController'
CALL_TIMEOUT_S = 10.0
CLOCK_MARGIN_S = 0.05     # sim time after the jump before the controllers restart (all nodes have the new time)
MOVE_S = 1.0              # time_from_start of the move back to the initial joint values
HOLD_S = 0.5              # sim time the joints have to stay at their initial values
SETTLE_TIMEOUT_S = 20.0   # wall
TOL = {'revolute': math.radians(1.0), 'continuous': math.radians(1.0), 'prismatic': 0.001}


def initial_values(urdf):
    """{joint: (position initial_value, tolerance)} of the ros2_control joints."""
    root = ET.fromstring(urdf)
    types = {j.get('name'): j.get('type') for j in root.findall('joint')}
    out = {}
    for j in root.iter('ros2_control'):
        for jt in j.findall('joint'):
            for si in jt.findall('state_interface'):
                v = si.find("param[@name='initial_value']")
                if si.get('name') == 'position' and v is not None:
                    out[jt.get('name')] = (float(v.text), TOL.get(types.get(jt.get('name')), TOL['revolute']))
    return out


class Reset(Node):
    def __init__(self):
        super().__init__('simulator')
        world = self.declare_parameter('world', '').value
        group = ReentrantCallbackGroup()   # the service calls are made from the reset service's callback
        self.list = self.create_client(ListControllers, '/controller_manager/list_controllers', callback_group=group)
        self.switch = self.create_client(SwitchController, '/controller_manager/switch_controller',
                                         callback_group=group)
        self.control = self.create_client(ControlWorld, f'/world/{world}/control', callback_group=group)
        self.lock = threading.Lock()
        self.q, self.evaluator_resets, self.initial = {}, None, {}
        self.create_subscription(JointState, '/joint_states', self.on_joints, 10, callback_group=group)
        self.create_subscription(String, '/evaluator/state', self.on_evaluator, 10, callback_group=group)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/robot_description', self.on_description, latched, callback_group=group)
        self.pubs = {}
        self.busy = threading.Lock()
        self.create_service(Trigger, '~/reset', self.on_reset, callback_group=group)

    def on_joints(self, msg):
        with self.lock:
            self.q.update(zip(msg.name, msg.position))

    def on_evaluator(self, msg):
        self.evaluator_resets = json.loads(msg.data)['resets']

    def on_description(self, msg):
        self.initial = initial_values(msg.data)

    def publisher(self, topic, typ):
        if topic not in self.pubs:     # (a message published before discovery matched the controller is lost)
            self.pubs[topic] = pub = self.create_publisher(typ, topic, 10)
            self.wait(f'a subscriber on {topic}', lambda: pub.get_subscription_count() > 0, 5.0)
        return self.pubs[topic]

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def call(self, client, req):
        if not client.wait_for_service(timeout_sec=CALL_TIMEOUT_S):
            raise RuntimeError(f'{client.srv_name} not available')
        res = client.call(req, timeout_sec=CALL_TIMEOUT_S)
        if res is None:
            raise RuntimeError(f'{client.srv_name}: no response in {CALL_TIMEOUT_S} s')
        return res

    def switch_controllers(self, activate=(), deactivate=()):
        req = SwitchController.Request(activate_controllers=list(activate), deactivate_controllers=list(deactivate),
                                       strictness=SwitchController.Request.STRICT)
        if not self.call(self.switch, req).ok:
            raise RuntimeError(f'switch_controller failed (activate {activate}, deactivate {deactivate})')

    def wait(self, what, cond, timeout_s=SETTLE_TIMEOUT_S):
        end = time.monotonic() + timeout_s
        while not cond():
            if time.monotonic() > end:
                raise RuntimeError(f'timed out after {timeout_s} s waiting for {what}')
            time.sleep(0.005)

    def on_reset(self, req, res):
        if not self.busy.acquire(blocking=False):
            res.success, res.message = False, 'a reset is running'
            return res
        t0 = time.monotonic()
        try:
            self.reset()
            res.success, res.message = True, f'reset in {time.monotonic() - t0:.2f} s, sim time {self.now():.3f}'
        except RuntimeError as e:
            res.success, res.message = False, str(e)
        finally:
            self.busy.release()
        self.get_logger().info(f'reset: {res.message}' if res.success else f'reset FAILED: {res.message}')
        return res

    def reset(self):
        if not self.initial:
            raise RuntimeError('no /robot_description yet')
        controllers = self.call(self.list, ListControllers.Request()).controller
        active = [c.name for c in controllers if c.state == 'active']
        jtcs = {c.name: [i.split('/')[0] for i in c.required_command_interfaces] for c in controllers
                if c.type == JTC and c.name in active}
        bases = [c.name for c in controllers if c.type == OMNI and c.name in active]
        for name in bases:
            self.publisher(f'/{name}/cmd_vel', TwistStamped)
        for name in jtcs:
            self.publisher(f'/{name}/joint_trajectory', JointTrajectory)
        before, resets = self.now(), self.evaluator_resets
        self.switch_controllers(deactivate=active)
        req = ControlWorld.Request()
        req.world_control.reset.all = True
        if not self.call(self.control, req).success:
            raise RuntimeError('gz world reset refused')
        # the clock jumps back to the start; the controllers restarted before every node has the new time would see
        # a negative period on their next update
        self.wait('/clock to jump back', lambda: self.now() < before)
        jumped = self.now()
        self.wait('sim time to run after the reset', lambda: self.now() >= jumped + CLOCK_MARGIN_S)
        self.switch_controllers(activate=active)
        with self.lock:
            self.q.clear()        # joint states from before the reset do not count
        for name in bases:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            self.publisher(f'/{name}/cmd_vel', TwistStamped).publish(msg)
        targets = {}
        for name, joints in jtcs.items():
            q0 = {j: self.initial[j] for j in joints if j in self.initial}
            if not q0:
                continue
            pub = self.publisher(f'/{name}/joint_trajectory', JointTrajectory)
            point = JointTrajectoryPoint(positions=[v for v, _ in q0.values()])
            point.time_from_start.sec, point.time_from_start.nanosec = int(MOVE_S), int(MOVE_S % 1 * 1e9)
            pub.publish(JointTrajectory(joint_names=list(q0), points=[point]))
            targets.update(q0)

        moved, since = self.now() + MOVE_S, [None]

        def at_start():
            with self.lock:
                ok = all(j in self.q and abs(self.q[j] - v) <= tol for j, (v, tol) in targets.items())
            t = self.now()
            since[0] = (since[0] or t) if ok and t >= moved else None
            return since[0] is not None and t - since[0] >= HOLD_S
        self.wait('the joints at their initial values', at_start)
        if resets is not None:   # with a task
            self.wait('the evaluator to see the reset', lambda: self.evaluator_resets > resets)


def main():
    rclpy.init()
    node = Reset()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        ex.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
