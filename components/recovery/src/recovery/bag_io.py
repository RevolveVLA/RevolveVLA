"""The bag reader of the recovery scan: rosbag2_py over an episode's mcap bag, decoded into the messages supervision's
`BagSource` takes. Importable only where ROS 2 is (the simulator image): `scan_client` imports it there; the host never does.
Not exercised by the CPU tests (they pass a fake reader): it runs against real bags in the GPU acceptance.

Times are sim seconds: the bag is recorded on sim time, so the receive stamp is the sim clock."""
from __future__ import annotations

import math
from typing import Callable, Iterator

from revolve_common.supervision import BagMessage
from revolve_common.supervision.sources import JOINTS, RESULT, STATE


def decode(topic: str, msg) -> object:
    """A deserialised message -> BagMessage.data (see BagMessage)."""
    if topic in (STATE, RESULT):
        return msg.data                                         # std_msgs/String JSON
    if topic == JOINTS or topic.endswith("/joint_state"):       # sensor_msgs/JointState
        return {"name": list(msg.name), "position": list(msg.position)}
    robot = topic.split("/")[2]                                 # /model/<robot>/pose: tf2_msgs/TFMessage
    for tf in msg.transforms:
        if tf.child_frame_id == robot:
            q, p = tf.transform.rotation, tf.transform.translation
            return (p.x, p.y, 2.0 * math.atan2(q.z, q.w))
    return None


def read_messages(bag: str, wanted: Callable[[str], bool]) -> Iterator[BagMessage]:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"), rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types() if wanted(t.name)}
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(types)))
    classes = {name: get_message(kind) for name, kind in types.items()}
    while reader.has_next():
        topic, raw, stamp_ns = reader.read_next()
        data = decode(topic, deserialize_message(raw, classes[topic]))
        if data is not None:
            yield BagMessage(topic, stamp_ns * 1e-9, data)
