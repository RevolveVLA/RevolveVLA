"""rollout sim: the policy on the new simulator (DATAGEN-v2.md 4). Per config (scene instance) ONE sim (SimSession,
`--no-autostart`, `<set>/<seed>_sim/` as its `--out`, a leased container name + ROS domain), kept for the config's rollouts
(`release` removes it): every episode runs in a client container beside it (`python3 -m evaluation.rollout_client`: reset the
sim unless it is fresh, start the episode, stream the policy's actions, wait for the result). Retries are new episodes from
the start (the same reset). A SimFault (clock fault, sim exit, ready / reset / result timeout) relaunches the sim (up to
`infra_retries` per config) and runs the episode again: infrastructure faults are not failed episodes.

The record is the evaluator's verdict (DATAGEN 4.3): passed, phases {name: status}, failed_phase, reason, duration_s, episode,
bag, result_file; `error` when there is none: `wall limit` (the client ran over `wall_s`), `interrupted`, `infrastructure`, or
the client's failure. Params (yaml `rollout: {plugin: sim, ...}`): `record` (images | no-images, default images: a policy needs
the cameras, the oracle and the expert do not), `wall_s` (wall seconds per episode, 600), `ready_s` (wall seconds to be ready, 180),
`infra_retries` (3), `lead_s` (0.1: time_from_start of each streamed point). Limits of the episode itself (budget, phase
timeouts) are the task's. Only policies that bring their own actions (`client_only`: the oracle, the expert) run: a served checkpoint is
blocked on the new simulator."""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

from datagen.core.spec import SpecError
from revolve_common.sim_session import (REPO, DomainPool, RelaunchPolicy, SimFault, SimInfraError, SimSession,
                                        stop_sims)

from ...core.errors import EvalError
from ...core.interfaces import Rollout
from ...core.registry import REGISTRY
from ...core.sims import sim_prefix
from ...core.termination import LIMITS

PARAMS = ("record", "wall_s", "ready_s", "infra_retries", "lead_s")
CLIENT_SRC = ("/repo/components/evaluation/src", "/repo/components/datagen/src")   # in the client container (repo = /repo)
BLOCKED = ("policy {name!r} is not available on the new simulator: a served checkpoint needs a v2 dataset, a retrained "
           "checkpoint and a Rosetta contract for the new bag topics (use `policy: {{plugin: oracle}}` or `expert`)")


class Failed(Exception):
    """An episode that produced no verdict: the message becomes the record's `error`."""


def say(message: str) -> None:
    print(f"evaluate: {message}", file=sys.stderr)


def in_container(arg: str) -> str:
    """A host path inside the repo as the client container sees it (the repo is mounted at /repo); other args as they are."""
    p = Path(arg)
    return str(Path("/repo") / p.relative_to(REPO)) if p.is_absolute() and p.is_relative_to(REPO) else arg


class Scene:
    """The sim of one config, kept for its rollouts."""

    def __init__(self, session, name: str, retries: int):
        self.session, self.name, self.faults = session, name, RelaunchPolicy(retries)
        self.lock, self.running, self.fresh = threading.Lock(), False, True

    def launch(self) -> None:
        self.session.start()
        self.running = self.fresh = True

    def stop(self) -> None:
        self.session.close()
        self.running = False


@REGISTRY.register("rollout", "sim")
class SimRollout(Rollout):
    def __init__(self, spec, params):
        if unknown := sorted(set(params) - set(PARAMS)):
            raise EvalError(f"rollout sim: unknown parameters {unknown} (known: {list(PARAMS)})")
        self.spec, self.record = spec, params.get("record", "images")
        if self.record not in ("images", "no-images"):
            raise EvalError(f"rollout sim: record must be images or no-images, not {self.record!r}")
        limits = {k: float(params.get(k) or v) for k, v in LIMITS.items()}
        self.wall_s, self.ready_s = limits["wall_s"], limits["ready_s"]
        self.infra_retries = int(params.get("infra_retries", 3))
        self.lead_s = float(params.get("lead_s", 0.1))
        self.prefix, self.scenes = sim_prefix(spec), {}
        self.lock, self.stopped = threading.Lock(), threading.Event()
        self.pool = self.domain_pool()

    # ---- seams: what the CPU tests replace
    def domain_pool(self) -> DomainPool:
        return DomainPool(self.prefix)

    def session(self, item, sim_out: Path, name: str, domain: int, gui: bool):
        return SimSession(item.scene, self.spec.train.robot, self.spec.train.task, sim_out, name, domain, self.record,
                          gui=gui, log=sim_out.parent / f"{sim_out.name}.log")

    def run_client(self, session, rollout: Path, args: list[str], reset: bool, log: Path) -> int:
        """The policy client of one episode in a container beside the sim -> its exit code (a SimFault for a sim that is
        broken, TimeoutExpired over `wall_s`)."""
        command = ["python3", "-m", "evaluation.rollout_client", "--sim-out", str(session.out), "--rollout", str(rollout),
                   *map(in_container, args), "--ready-s", str(self.ready_s), "--lead-s", str(self.lead_s),
                   *(["--reset"] if reset else [])]
        client = session.client(command, pythonpath=CLIENT_SRC, mounts=[rollout])
        return client.run(log, timeout=self.wall_s, watch=lambda: None if self.stopped.is_set() else session.check())

    # ---- Rollout
    def validate(self, policy) -> None:
        if not policy.client_only:
            raise SpecError(BLOCKED.format(name=policy.name))

    def run(self, item, out, *, seed, runner, policy, ckpt, gui=False):
        out = Path(out).resolve()
        out.mkdir(parents=True, exist_ok=True)
        base = dict(seed=seed, dir=str(out), policy=policy.name, ckpt=ckpt.path if ckpt else None, scene=str(item.scene))
        try:
            episode = self._episode(item, out, policy, gui)
            res = episode["result"]
            record = dict(base, episode=episode["episode"], resets=episode["resets"], steps=episode.get("steps"),
                          phases={p["name"]: p["status"] for p in res["phases"]}, failed_phase=res["failed_phase"],
                          reason=res["reason"], check=res["check"], passed=bool(res["passed"]), duration_s=res["duration_s"],
                          bag=res["bag"], result_file=res["result_file"], error=None)
        except Failed as e:
            record = dict(base, episode=None, phases=None, failed_phase=None, reason=None, passed=False, error=str(e))
        (out / "rollout.json").write_text(json.dumps(record))
        return record

    def release(self, item) -> None:
        with self.lock:
            scene = self.scenes.pop(str(item.scene), None)
        if scene:
            scene.stop()
            self.pool.release(scene.name)

    def stop(self) -> None:
        self.stopped.set()
        stop_sims(self.prefix, self.pool.docker)

    def default_workers(self, policy) -> int:
        return 1                                                     # a sim runs in real time on the GPU

    # ---- one episode
    def _scene(self, item, out: Path, gui: bool) -> Scene:
        with self.lock:
            if (key := str(item.scene)) not in self.scenes:
                name, domain = self.pool.lease()
                sim_out = out.parent / f"{item.config.seed}_sim"
                self.scenes[key] = Scene(self.session(item, sim_out, name, domain, gui), name, self.infra_retries)
            return self.scenes[key]

    def _episode(self, item, out: Path, policy, gui: bool) -> dict:
        args = policy.client_args(item)
        (out / "episode.json").unlink(missing_ok=True)
        scene = self._scene(item, out, gui)
        with scene.lock:
            while True:
                if self.stopped.is_set():
                    raise Failed("interrupted: the run was stopped")
                try:
                    if not scene.running:
                        scene.launch()
                    rc = self.run_client(scene.session, out, args, not scene.fresh, out / "client.log")
                except SimFault as fault:
                    scene.stop()
                    try:
                        scene.faults.record(fault)
                    except SimInfraError as e:
                        raise Failed(f"infrastructure: {e}") from e
                    say(f"sim {scene.name}: {fault}: relaunch {len(scene.faults.faults)}/{scene.faults.max_retries}")
                    continue
                except subprocess.TimeoutExpired:
                    scene.stop()                                   # the episode may still run: the next one gets a fresh sim
                    raise Failed(f"wall limit {self.wall_s:.0f} s: the policy client did not finish; see {out / 'client.log'}")
                scene.fresh = False
                if rc != 0:
                    scene.stop()
                    raise Failed("interrupted: the client was killed" if self.stopped.is_set() else
                                 f"the policy client exited with {rc}; see {out / 'client.log'}")
                try:
                    return json.loads((out / "episode.json").read_text())
                except (OSError, ValueError) as e:
                    raise Failed(f"the policy client left no episode.json ({e}); see {out / 'client.log'}") from e
