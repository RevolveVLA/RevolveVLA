"""What Rosetta's porter does with one bag's time line, computed from the bag itself, in the dataset image (rosbag2_py,
Rosetta): the frames it must emit (the exact frame check, checks.py) and how many of them carry a zero-filled action.

The porter (rosetta_ros/offline/bag_frames.py) lays ticks 1/fps apart from the bag's starting_time over its duration,
n = duration // step + 1, and skips every tick before each observation stream has a sample: a tick t is emitted once
every observation topic has a message received at or before t. So warmup = the ticks strictly before the latest of
the observation topics' first receive times. A dataset episode with any other frame count lost or gained frames.
Action streams do not gate warmup: an emitted frame whose tick comes before an action topic's first message holds
zeros for that action (late_action_frames)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Ticks:
    ticks: int
    warmup: int
    action_late: dict = field(default_factory=dict)     # {action topic: emitted frames before its first message}

    @property
    def frames(self) -> int:
        return self.ticks - self.warmup


def topics(contract_path) -> tuple[list[str], list[str]]:
    """(observation topics, action topics) of a contract, read with Rosetta's own loader and spec iterators, as the
    porter reads them."""
    from rosetta.contract.schema import load_contract
    from rosetta.contract.specs import iter_action_specs, iter_observation_specs
    c = load_contract(contract_path)
    return (sorted({s.source.channel.topic for s in iter_observation_specs(c)}),
            sorted({s.source.channel.topic for s in iter_action_specs(c)}))


def ticks_before(start_ns: int, step_ns: int, t_ns: int) -> int:
    """The number of ticks start + i * step (i >= 0) strictly before t."""
    return 0 if t_ns <= start_ns else -(-(t_ns - start_ns) // step_ns)


def late_action_frames(start_ns: int, step_ns: int, warmup: int, first_ns: int) -> int:
    """Emitted frames (ticks i >= warmup) whose tick is strictly before an action topic's first receive time: they hold
    zeros for that action. A message received exactly at a tick counts for that tick."""
    return max(0, ticks_before(start_ns, step_ns, first_ns) - warmup)


def of_bag(bag: Path, obs_topics, fps: int, action_topics=()) -> Ticks:
    import rosbag2_py
    from rosetta.frames.resample import step_ns
    from rosetta_ros.bag_metadata import BAG_METADATA_KEY, read_bag_metadata

    storage = read_bag_metadata(bag).get(BAG_METADATA_KEY, {}).get("storage_identifier", "mcap")
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=storage),
                rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))
    wanted = set(obs_topics) | set(action_topics)
    try:
        meta = reader.get_metadata()
        start, step = meta.starting_time.nanoseconds, step_ns(fps)
        n = max(1, meta.duration.nanoseconds // step + 1)
        reader.set_filter(rosbag2_py.StorageFilter(topics=sorted(wanted)))
        first = {}
        while len(first) < len(wanted) and reader.has_next():
            topic, _, t = reader.read_next()
            first.setdefault(topic, t)
    finally:
        reader.close()
    if missing := sorted(wanted - set(first)):
        # metadata.yaml counts messages on them (manifest.check_inputs), so the data itself is short or damaged
        raise ValueError(f"no message could be read on {missing}")
    warmup = min(n, ticks_before(start, step, max(first[t] for t in obs_topics)))
    late = {t: min(n - warmup, late_action_frames(start, step, warmup, first[t])) for t in action_topics}
    return Ticks(n, warmup, late)


def of_episodes(episodes, contract_path, fps: int) -> tuple[list[Ticks], list[str]]:
    """Every episode's bag (manifest.Episode) -> (its Ticks in order, problems). An input check, before the port: a bag
    rosbag2 cannot read is an `inputs:` problem."""
    obs, actions = topics(contract_path)
    out, problems = [], []
    for e in episodes:
        try:
            out.append(of_bag(e.bag, obs, fps, actions))
        except Exception as ex:      # rosbag2 raises RuntimeError and friends for a missing or damaged file
            problems.append(f"inputs: {e.where}: bag {e.bag} unreadable by rosbag2 ({type(ex).__name__}: {ex})")
    return out, problems
