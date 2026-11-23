"""Host side of one running simulator (components/datagen/DATAGEN-v2.md section 4): `./revolve sim` as a subprocess (4.1),
the client container beside it (4.5), the faults that mean "relaunch" (4.2, 4.4, 12). No rclpy and no docker SDK: the
docker CLI through `Docker`, so the datagen and evaluation hosts import it freely. The client side is sim_client.py
(runs inside the container, imports this module for SimFault / wait_files); sim_fake.py is the CPU stand-in.
"""
from __future__ import annotations

import itertools
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SIM_IMAGE = os.environ.get("REVOLVE_SIM_IMAGE", "revolve-simulator:latest")
READY_TIMEOUT_S = 180
NO_CLOCK, NO_CLOCK_LINES = "No clock received", 3       # a broken launch (DATAGEN section 12); 1-2 lines are a race
DOMAINS = range(102, 233)                               # revolve sim derives its own from container names in 1..101

# Faults: the sim or its wiring is broken, not the episode. exit code of a client that hit one (host: SimFault.from_exit)
READY_TIMEOUT, RESET_FAILED, RESULT_TIMEOUT, FILES_TIMEOUT = "ready_timeout", "reset_failed", "result_timeout", \
    "files_timeout"
SIM_EXIT, CLOCK_FAULT = "sim_exit", "clock_fault"       # seen by the host only
EXIT_CODES = {READY_TIMEOUT: 75, RESET_FAILED: 76, RESULT_TIMEOUT: 77, FILES_TIMEOUT: 78}


class SimFault(Exception):
    """An infrastructure fault of the sim: the caller closes the session and relaunches it."""

    def __init__(self, kind: str, detail: str = ""):
        self.kind, self.detail = kind, detail
        super().__init__(f"{kind}: {detail}" if detail else kind)

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.kind]

    @classmethod
    def from_exit(cls, rc: int, detail: str = "") -> SimFault | None:
        kind = next((k for k, c in EXIT_CODES.items() if c == rc), None)
        return cls(kind, detail) if kind else None


class SimInfraError(RuntimeError):
    """The infrastructure retries ran out (or no worker slot is free)."""


class Docker:
    """The docker CLI calls a session needs; tests pass a fake with the same methods."""

    def _run(self, *args) -> str:
        try:
            r = subprocess.run(["docker", *args], capture_output=True, text=True)
        except FileNotFoundError:                       # no docker: nothing runs
            return ""
        return r.stdout

    def rm(self, name: str) -> None:
        self._run("rm", "-f", name)

    def names(self) -> list[str]:
        return self._run("ps", "--format", "{{.Names}}").split()

    def domains(self) -> set[int]:
        """ROS_DOMAIN_ID of every running container (they share the host network)."""
        ids = self._run("ps", "-q").split()
        env = self._run("inspect", "--format", "{{range .Config.Env}}{{println .}}{{end}}", *ids) if ids else ""
        return {int(m) for m in re.findall(r"^ROS_DOMAIN_ID=(\d+)$", env, re.M)}


class DomainPool:
    """Distinct (container name, ROS domain) per worker: names `<prefix>-<k>`, domains from `domains`, skipping what is
    leased here and what running containers use (a name or a domain). The prefix names the run."""

    def __init__(self, prefix: str, domains=DOMAINS, docker: Docker | None = None):
        self.prefix, self.domains, self.docker = prefix, list(domains), docker or Docker()
        self._leased: dict[str, int] = {}
        self._lock = threading.Lock()

    def lease(self) -> tuple[str, int]:
        with self._lock:
            names, used = set(self.docker.names()), set(self.docker.domains()) | set(self._leased.values())
            domain = next((d for d in self.domains if d not in used), None)
            if domain is None:
                raise SimInfraError(f"no free ROS_DOMAIN_ID in {self.domains[0]}..{self.domains[-1]}")
            name = next(n for n in (f"{self.prefix}-{k}" for k in itertools.count())
                        if n not in self._leased and n not in names and f"{n}-client" not in names)
            self._leased[name] = domain
            return name, domain

    def release(self, name: str) -> None:
        with self._lock:
            self._leased.pop(name, None)


def stop_sims(prefix: str, docker: Docker | None = None) -> int:
    """`docker rm -f` every running container named `<prefix>-...` (sims and their clients) -> how many. What an
    interrupted run leaves behind; never raises."""
    docker = docker or Docker()
    names = [n for n in docker.names() if n.startswith(prefix + "-")]
    for n in names:
        docker.rm(n)
    return len(names)


def wait_files(out, n: int, timeout_s: float = 30.0) -> dict:
    """The parsed `<out>/episode_<n>.json`: once it exists the episode's files are complete (DATAGEN 4.3)."""
    f = Path(out) / f"episode_{n}.json"
    end = time.monotonic() + timeout_s
    while True:
        try:
            return json.loads(f.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            if time.monotonic() > end:
                raise SimFault(FILES_TIMEOUT, f"no {f}") from None
            time.sleep(0.05)


class SimSession:
    """One sim (one scene instance) as a `revolve sim` subprocess. `start()` launches, `check()` says whether it is
    still sane, `close()` removes the container (idempotent). A context manager. `connect()` does not exist here: the
    client runs in a container (`client()`), see sim_client.py."""

    def __init__(self, scene, robot: str, task: str, out, name: str, domain_id: int, record: str = "images",
                 gui: bool = False, *, log=None, repo: Path = REPO, docker: Docker | None = None, popen=subprocess.Popen):
        if record not in ("images", "no-images"):
            raise ValueError(f"record must be images or no-images, not {record!r}")
        self.scene, self.robot, self.task, self.name, self.domain_id = str(scene), robot, task, name, domain_id
        self.out, self.record, self.gui, self.repo = Path(out).resolve(), record, gui, Path(repo)
        self.log = Path(log) if log else self.out.parent / f"{name}.log"
        self.docker, self._popen = docker or Docker(), popen
        self._proc, self._log_pos, self._no_clock = None, 0, 0

    def argv(self) -> list[str]:
        """The launch command, as DATAGEN 4.1 gives it (`--no-autostart`: episodes start on /evaluator/start)."""
        return [str(self.repo / "revolve"), "sim", "--scene", self.scene, "--robot", self.robot, "--task", self.task,
                "--no-autostart", "--record", self.record, "--out", str(self.out), "--name", self.name,
                "--domain-id", str(self.domain_id), *(["--gui"] if self.gui else [])]

    def start(self) -> None:
        """Launch the sim (the log is appended: a relaunch keeps the earlier launches). Does not wait for ready."""
        self.close()
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log, "a") as f:
            f.write(f"--- {time.strftime('%F %T')} {' '.join(self.argv())}\n")
            f.flush()
            self._log_pos, self._no_clock = f.tell(), 0
            self._proc = self._popen(self.argv(), stdout=f, stderr=subprocess.STDOUT, cwd=self.repo,
                                     start_new_session=True)

    def check(self) -> SimFault | None:
        """The fault of this launch, if any: the sim exited (`revolve sim` returns when the container is gone), or the
        controller manager never got /clock (NO_CLOCK_LINES log lines). None before start and after close."""
        if self._proc is None:
            return None
        if (rc := self._proc.poll()) is not None:
            return SimFault(SIM_EXIT, f"revolve sim exited with {rc}; see {self.log}")
        with open(self.log, "rb") as f:
            f.seek(self._log_pos)
            data = f.read()
            self._log_pos += len(data)
        self._no_clock += data.decode(errors="replace").count(NO_CLOCK)
        if self._no_clock >= NO_CLOCK_LINES:
            return SimFault(CLOCK_FAULT, f"{self._no_clock} {NO_CLOCK!r} lines; see {self.log}")
        return None

    def close(self) -> None:
        """`docker rm -f` the sim (this also ends the `revolve sim` process). Safe to call again."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        self.docker.rm(self.name)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()

    def client(self, command: list[str], **kw) -> ClientContainer:
        return ClientContainer(f"{self.name}-client", self.domain_id, command, self.out, docker=self.docker, **kw)

    def __enter__(self) -> SimSession:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class ClientContainer:
    """A client of the sim: `command` in a container from the sim image (ROS 2 + rclpy), as the simulator's own contract
    test starts one: host network, the sim's ROS_DOMAIN_ID, Fast DDS over UDPv4, the repo read-only at /repo (common on
    PYTHONPATH, plus `pythonpath`), the Kura and ros_gz installs, `out` and `mounts` (host dirs) writable at their own
    path. `run()` blocks until the command ends."""

    STOP_WAIT_S = 10            # after `docker rm -f`, before the local docker process is killed

    def __init__(self, name: str, domain_id: int, command: list[str], out, *, image: str = SIM_IMAGE,
                 repo: Path = REPO, pythonpath=(), mounts=(), docker: Docker | None = None):
        self.name, self.domain_id, self.command, self.image, self.repo = name, domain_id, list(command), image, Path(repo)
        self.mounts = [Path(out).resolve(), *map(Path, mounts)]
        self.pythonpath, self.docker = ["/repo/common", *pythonpath], docker or Docker()

    def argv(self) -> list[str]:
        r = self.repo
        return ["docker", "run", "--rm", "--name", self.name, "--network", "host", "-u", f"{os.getuid()}:{os.getgid()}",
                "-e", "HOME=/tmp", "-e", f"ROS_DOMAIN_ID={self.domain_id}", "-e", "FASTDDS_BUILTIN_TRANSPORTS=UDPv4",
                "-e", f"PYTHONPATH={':'.join(self.pythonpath)}", "-v", f"{r}:/repo:ro",
                *[a for m in self.mounts for a in ("-v", f"{m}:{m}")],
                "-v", f"{r}/ThirdParty/gz_kura/install:/opt/gz_ws/install:ro",
                "-v", f"{r}/ThirdParty/ros_gz_ws/install:/opt/ros_gz_ws/install:ro",
                self.image, "bash", "-c", 'exec "$@"', "client", *self.command]

    def run(self, log, *, timeout: float | None = None, watch=None, poll_s: float = 2.0) -> int:
        """Run to the end, stdout + stderr to `log` -> the exit code. `watch()` (e.g. `session.check`) is polled every
        `poll_s`: a SimFault it returns removes the client and is raised. A client that exits with one of EXIT_CODES
        raises that SimFault; `timeout` (wall s) raises subprocess.TimeoutExpired after removing the client."""
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        end = None if timeout is None else time.monotonic() + timeout
        with open(log, "ab") as f:
            proc = subprocess.Popen(self.argv(), stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                next_watch = 0.0
                while proc.poll() is None:
                    if watch and time.monotonic() >= next_watch:
                        next_watch = time.monotonic() + poll_s
                        if fault := watch():
                            raise fault
                    if end is not None and time.monotonic() > end:
                        raise subprocess.TimeoutExpired(self.argv(), timeout)
                    time.sleep(0.1)
            finally:
                if proc.poll() is None:
                    self.close()
                    try:
                        proc.wait(timeout=self.STOP_WAIT_S)
                    except subprocess.TimeoutExpired:
                        proc.kill()
        if fault := SimFault.from_exit(proc.returncode, f"client exit {proc.returncode}; see {log}"):
            raise fault
        return proc.returncode

    def close(self) -> None:
        self.docker.rm(self.name)


class RelaunchPolicy:
    """At most `max_retries` infrastructure retries (shared by the sessions that use it); `faults` lists them, for the
    manifest: an infrastructure retry is not a failed episode."""

    def __init__(self, max_retries: int = 3):
        self.max_retries, self.faults = max_retries, []

    def record(self, fault: SimFault) -> None:
        if len(self.faults) >= self.max_retries:
            raise SimInfraError(f"{fault}; {self.max_retries} infrastructure retries used: {self.faults}") from fault
        self.faults.append(f"{fault.kind}")


def run_with_relaunch(session, work, policy: RelaunchPolicy | None = None, log=print):
    """`session.start()`, then `work(session)`; a SimFault from it (a client's exit code, `session.check()` through
    `ClientContainer.run(watch=...)`) closes the session and relaunches it, counted in `policy`; SimInfraError when the
    retries run out. The session is closed at the end (also when `work` raises anything). -> work's value."""
    policy = policy or RelaunchPolicy()
    try:
        session.start()
        while True:
            try:
                return work(session)
            except SimFault as fault:
                policy.record(fault)
                log(f"sim {session.name}: {fault}: relaunch {len(policy.faults)}/{policy.max_retries}")
                session.start()                         # start() closes the failed launch first
    finally:
        session.close()
