"""The robots' Rosetta contracts (specs/robots/<r>/contract.yaml) with Rosetta's own code, in the simulator image (ROS 2
Lyrical: the message packages Rosetta's loader imports). Rosetta is not built for Lyrical anywhere in this repo; its
two pure-Python packages (rosetta, rosetta_ros) are pip-installed from the ThirdParty/rosetta submodule into the
container's /tmp first (conftest.ROSETTA_PIP), so that the `ros2` robot interface entry point is registered exactly
as a colcon build registers it. Prints one RESULT JSON line.

  python3 check_rosetta.py --robots R [R ...] [--bag BAG ...] [--task T]

Per robot: rosetta.contract.schema.load_contract (the whole validation: types, timelines, QoS, operators, layout),
the resolved streams (key, topic, timeline, width) and revolve_common.specs.check_contract with the cameras from the
robot's description (xacro). Per bag (the first robot's contract): every contract topic is in the bag with messages
and the contract's type; no zero header stamp on a header-timeline source; Rosetta's own decoder on the first,
middle and last message of every source (width, image size, finite); /task/prompt exactly once and the bag
metadata's `lerobot.operator_prompt` (the porter's task for frames before the first /task/prompt message), both equal
to task T's prompt; the times of the prompt message and of the last observation stream's first message (warmup end)
are reported."""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import rosbag2_py
import yaml
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from revolve_common import specs

REPO = Path('/repo')


def load(robot):
    from rosetta.contract.schema import load_contract
    from rosetta.contract.specs import iter_specs
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        c = load_contract(REPO / 'specs' / 'robots' / robot / specs.CONTRACT_FILE)
    return c, list(iter_specs(c)), [str(x.message) for x in w]


def describe(robot):
    c, streams, warns = load(robot)
    r = specs.load(REPO / 'specs' / 'robots' / robot / 'robot.yaml', assets=False)
    width = lambda key: sum(s.dim for s in streams if s.key == key)   # noqa: E731
    return {'robot_type': c.robot_type, 'fps': c.fps, 'tasks': [t.channel.topic for t in c.tasks],
            'streams': [[s.key, s.source.channel.topic, s.source.align.timeline, s.dim, s.dtype,
                         list(s.image_resize) if getattr(s, 'image_resize', None) else None] for s in streams],
            'state': width('observation.state'), 'action': width('action'),
            'images': sorted(s.key for s in streams if s.is_image), 'warnings': warns,
            'check_contract': specs.check_contract(r, REPO / 'specs' / 'robots' / robot)}


def stamp_ns(m):
    return m.header.stamp.sec * 10**9 + m.header.stamp.nanosec


def check_bag(robot, bag, prompt):
    from rosetta.frames.codecs import decode_value
    c, streams, _ = load(robot)
    topics = {s.source.channel.topic: s.source.channel.type for s in streams}
    topics.update({t.channel.topic: t.channel.type for t in c.tasks})
    header = {s.source.channel.topic for s in streams if s.source.align.timeline == 'header'}
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    msgs, first_t, zero = {t: [] for t in topics}, {}, {t: 0 for t in topics}
    while r.has_next():
        topic, data, t = r.read_next()
        if topic not in topics:
            continue
        m = deserialize_message(data, get_message(types[topic]))
        first_t.setdefault(topic, t)
        if hasattr(m, 'header'):
            zero[topic] += stamp_ns(m) == 0
        msgs[topic].append(m)
    out = {'counts': {t: len(v) for t, v in msgs.items()},
           'missing': sorted(t for t in topics if not msgs[t]),
           'type_mismatch': sorted(t for t in topics if t in types and types[t] != topics[t]),
           'zero_header_stamps': {t: zero[t] for t in sorted(header) if zero[t]},
           'zero_stamps_receive_timeline': {t: zero[t] for t in sorted(set(topics) - header) if zero[t]},
           'decode': {}, 'decode_problems': []}
    for s in streams:
        ms = msgs[s.source.channel.topic]
        if not ms:
            continue
        shapes = []
        for m in (ms[0], ms[len(ms) // 2], ms[-1]):
            try:
                v = np.asarray(decode_value(m, s))
            except Exception as e:      # Rosetta raises ValueError and friends per message
                out['decode_problems'].append(f'{s.key} {s.source.channel.topic}: {type(e).__name__}: {e}')
                continue
            shapes.append(list(v.shape))
            want = [*s.image_resize, 3] if s.is_image else [s.dim]
            if list(v.shape) != want or (not s.is_image and not np.isfinite(v).all()):
                out['decode_problems'].append(f'{s.key} {s.source.channel.topic}: shape {list(v.shape)} '
                                              f'(want {want}) or not finite: {v.tolist() if not s.is_image else ""}')
        out['decode'][f'{s.key} {s.source.channel.topic}'] = shapes
    tasks = [m for t in c.tasks for m in msgs[t.channel.topic]]
    out['prompts'] = [m.data for m in tasks]
    meta = yaml.safe_load((Path(bag) / 'metadata.yaml').read_text())['rosbag2_bagfile_information']
    out['metadata_prompt'] = (meta.get('custom_data') or {}).get('lerobot.operator_prompt')   # Rosetta's fallback
    obs = [s.source.channel.topic for s in streams if s.key.startswith('observation.')]
    if tasks and all(t in first_t for t in obs):
        out['prompt_t'] = first_t[c.tasks[0].channel.topic] * 1e-9
        out['warmup_t'] = max(first_t[t] for t in obs) * 1e-9      # the last observation stream's first message
        # the frames Rosetta's porter emits: ticks 1/fps apart over the bag's span, minus those before warmup
        start, step = meta['starting_time']['nanoseconds_since_epoch'], 10**9 // c.fps
        ticks = meta['duration']['nanoseconds'] // step + 1
        out['frames_est'] = int(ticks - sum(start + k * step < max(first_t[t] for t in obs) for k in range(ticks)))
        out['bag_span_s'] = [start * 1e-9, (start + meta['duration']['nanoseconds']) * 1e-9]
    out['ok'] = (not out['missing'] and not out['type_mismatch'] and not out['zero_header_stamps']
                 and not out['decode_problems'] and out['prompts'] == [prompt] and out['metadata_prompt'] == prompt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--robots', nargs='+', required=True)
    ap.add_argument('--bag', nargs='*', default=[])
    ap.add_argument('--task')
    a = ap.parse_args()
    out = {'contracts': {r: describe(r) for r in a.robots}, 'bags': {}}
    if a.bag:
        prompt = specs.load(REPO / 'specs' / 'tasks' / a.task / 'task.yaml', assets=False).prompt
        out['prompt'] = prompt
        out['bags'] = {b: check_bag(a.robots[0], b, prompt) for b in a.bag}
    print('RESULT ' + json.dumps(out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
