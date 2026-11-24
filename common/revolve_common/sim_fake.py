"""CPU stand-ins for sim_session.py + sim_client.py: `FakeSession` has SimSession's constructor and methods, `FakeClient`
SimClient's loop (wait_ready, reset, start_episode, wait_result, wait_files). No docker, no ROS: each episode writes
`episode_<n>.json` (the DATAGEN 4.3 schema) and an empty `episode_<n>_bag/` into `out`, numbered on from the highest
file there, as the sim does. `faults` = {op: [kind, ...]}: the next call of that op (wait_ready, reset, wait_result,
check) raises / returns that SimFault kind once, shared by the relaunches of the session. `reaches(target)` says whether
a warm reset (`reset_to`) gets to its target."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from .sim_session import SimFault
from .sim_client import EpisodeRefused
from .warm_reset import ResetReport

PHASES = ("approach", "grasp", "open")


def fake_result(task: str, scene: str, robot: str, episode: int, passed: bool, out) -> dict:
    """An evaluator result as DATAGEN 4.3 gives it; a failed one fails the first phase on its timeout."""
    out = Path(out).resolve()
    phases = [{"name": p, "status": "passed" if passed else ("failed" if i == 0 else "not_reached"),
               "t_start": None if (i and not passed) else float(i), "t_end": None if (i and not passed) else i + 1.0,
               "reason": "timeout_s" if (not passed and i == 0) else None, "last": {}} for i, p in enumerate(PHASES)]
    return {"task": task, "scene": scene, "robot": robot, "episode": episode, "sim_time_start": 3.9,
            "passed": passed, "failed_phase": None if passed else PHASES[0], "reason": None if passed else "timeout_s",
            "t_fail": None if passed else 20.0, "check": None, "condition": None, "value": None,
            "duration_s": float(len(PHASES)) if passed else 20.0, "phases": phases, "sanity_last": {},
            "rate_hz": 99.0, "tick_ms": {"mean": 1.0, "max": 2.0}, "contacts": [], "gz_log": None,
            "result_file": str(out / f"episode_{episode}.json"), "bag": str(out / f"episode_{episode}_bag")}


def scene_name(scene: str) -> str:
    p = Path(scene)
    return (yaml.safe_load(p.read_text()) or {}).get("name", p.parent.name) if p.is_file() else scene


class FakeSession:
    def __init__(self, scene, robot: str, task: str, out, name: str, domain_id: int, record: str = "images",
                 gui: bool = False, *, passes=lambda n: True, bagless=lambda n: False,
                 faults: dict | None = None, reaches=lambda target: True, **_):
        self.scene, self.robot, self.task, self.name, self.domain_id = str(scene), robot, task, name, domain_id
        self.out, self.record, self.gui = Path(out).resolve(), record, gui
        self.passes, self.bagless, self.reaches = passes, bagless, reaches
        self.faults = {k: list(v) for k, v in (faults or {}).items()}
        self.warm_targets = []
        self.started, self.launches, self.closed = False, 0, 0
        self.client_state = {"resets": 0, "next": 0}

    def fault(self, op: str) -> None:
        if self.faults.get(op):
            raise SimFault(self.faults[op].pop(0), "injected")

    def start(self) -> None:
        self.close()
        self.out.mkdir(parents=True, exist_ok=True)
        self.started, self.launches = True, self.launches + 1
        done = [int(f.stem.split("_")[1]) for f in self.out.glob("episode_*.json") if f.stem.split("_")[1].isdigit()]    # as the evaluator's
        self.client_state = {"resets": 0, "next": max(done) + 1 if done else 0}

    def check(self) -> SimFault | None:
        if self.started and self.faults.get("check"):
            return SimFault(self.faults["check"].pop(0), "injected")
        return None

    def close(self) -> None:
        if self.started:
            self.started, self.closed = False, self.closed + 1

    def connect(self) -> FakeClient:
        return FakeClient(self)

    def __enter__(self) -> FakeSession:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class FakeClient:
    def __init__(self, session: FakeSession):
        self.s, self.out, self._running = session, session.out, None

    def wait_ready(self, timeout_s: float = 0) -> None:
        self.s.fault("wait_ready")

    def reset(self, timeout_s: float = 0) -> None:
        self.s.fault("reset")
        self.s.client_state["resets"] += 1

    def reset_to(self, target, tol=None, timeout_s: float = 0) -> ResetReport:
        self.reset()
        self.s.warm_targets.append(target)
        if self.s.reaches(target):
            return ResetReport(True, 0.01, 0.01, 0.01, wall_s=0.1, sim_s=0.1)
        return ResetReport(False, 0.5, 0.0, 0.0, wall_s=0.1, sim_s=0.1, reason="missed the target (fake)")

    def resets(self) -> int:
        return self.s.client_state["resets"]

    def start_episode(self) -> int:
        if self._running is not None:
            raise EpisodeRefused(f"episode {self._running} is running")
        self._running, self.s.client_state["next"] = self.s.client_state["next"], self.s.client_state["next"] + 1
        return self._running

    def has_result(self, n: int) -> bool:
        return True                                 # an episode of the fake ends at once

    def wait_result(self, n: int, timeout_s: float = 0) -> dict:
        self.s.fault("wait_result")
        s = self.s
        res = fake_result(s.task, scene_name(s.scene), s.robot, n, bool(s.passes(n)), s.out)
        if s.bagless(n):                            # the recorder gave none
            res["bag"] = None
        else:
            (s.out / f"episode_{n}_bag").mkdir(exist_ok=True)
        (s.out / f"episode_{n}.json").write_text(json.dumps(res))
        self._running = None
        return res

    def wait_files(self, n: int, timeout_s: float = 0) -> dict:
        return json.loads((self.out / f"episode_{n}.json").read_text())

    def now(self) -> float:
        return 1.0

    def close(self) -> None:
        pass
