"""Host measurements of the capacity bench: a sampler thread (GPU, CPU, RAM, load, GPU memory per container) and the timing of every
client container run. Pure parsers on text, so the CPU tests need no GPU."""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from datetime import datetime
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from statistics import mean

CONTAINER_ID = re.compile(r"[0-9a-f]{64}")


def parse_gpu(text: str) -> tuple[float, float]:
    """`nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits` -> (util %, used MiB)."""
    util, used = (float(x) for x in text.strip().splitlines()[0].split(","))
    return util, used


def parse_apps(text: str) -> dict[int, float]:
    """`nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits` -> {pid: MiB}."""
    apps = {}
    for line in text.strip().splitlines():
        pid, used = (x.strip() for x in line.split(","))
        apps[int(pid)] = float(used)
    return apps


def cpu_busy(stat_line: str) -> tuple[float, float]:
    """The first line of /proc/stat -> (busy jiffies, total jiffies); idle and iowait count as idle."""
    v = [float(x) for x in stat_line.split()[1:]]
    idle = v[3] + (v[4] if len(v) > 4 else 0.0)
    return sum(v[:8]) - idle, sum(v[:8])


def cpu_percent(before: tuple[float, float], after: tuple[float, float]) -> float:
    total = after[1] - before[1]
    return 100.0 * (after[0] - before[0]) / total if total > 0 else 0.0


def ram_used_gb(meminfo: str) -> float:
    kb = {k: float(v.split()[0]) for k, v in (ln.split(":", 1) for ln in meminfo.splitlines() if ":" in ln)}
    return (kb["MemTotal"] - kb["MemAvailable"]) / 1048576


def container_id(cgroup: str) -> str | None:
    """The docker container id in a /proc/<pid>/cgroup text (cgroup v1 and v2 layouts)."""
    found = CONTAINER_ID.search(cgroup)
    return found.group(0) if found else None


def gpu_by_container(apps: dict[int, float], cgroups: dict[int, str], names: dict[str, str]) -> dict[str, float]:
    """GPU MiB per container name (processes of no container: key `host`)."""
    out: dict[str, float] = defaultdict(float)
    for pid, mib in apps.items():
        cid = container_id(cgroups.get(pid, ""))
        out[names.get(cid, "host") if cid else "host"] += mib
    return dict(out)


def _run(*argv: str) -> str:
    return subprocess.run(argv, capture_output=True, text=True, timeout=20).stdout


def _read(path: str) -> str:
    try:
        return Path(path).read_text()
    except OSError:
        return ""


class Sampler(threading.Thread):
    """Every `period_s`: GPU util + used MiB, GPU MiB per container, CPU %, RAM GB, load 1 min -> `samples` (dicts)."""

    def __init__(self, period_s: float = 2.0, run=_run, read=_read, clock=time.monotonic):
        super().__init__(daemon=True)
        self.period_s, self.cmd, self.read, self.clock = period_s, run, read, clock
        self.samples: list[dict] = []
        self.stop_event = threading.Event()

    def sample(self, prev: tuple[float, float]) -> tuple[dict, tuple[float, float]]:
        busy = cpu_busy(self.read("/proc/stat").splitlines()[0])
        util, used = parse_gpu(self.cmd("nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"))
        apps = parse_apps(self.cmd("nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"))
        names = {i: n for i, n in (ln.split(maxsplit=1) for ln in self.cmd("docker", "ps", "--no-trunc", "--format", "{{.ID}} {{.Names}}").splitlines())}
        return ({"t": self.clock(), "gpu_util": util, "gpu_mib": used, "cpu": cpu_percent(prev, busy),
                 "ram_gb": ram_used_gb(self.read("/proc/meminfo")), "load1": os.getloadavg()[0],
                 "containers": gpu_by_container(apps, {p: self.read(f"/proc/{p}/cgroup") for p in apps}, names),
                 "running": sorted(names.values())}, busy)

    def run(self) -> None:
        prev = cpu_busy(self.read("/proc/stat").splitlines()[0])
        while not self.stop_event.wait(self.period_s):
            try:
                one, prev = self.sample(prev)
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                continue
            self.samples.append(one)

    def stop(self) -> list[dict]:
        self.stop_event.set()
        self.join(timeout=30)
        return self.samples


def summarise(samples: list[dict], prefixes: tuple[str, ...], baseline_mib: float) -> dict:
    """Peaks and means of the samples; a slot = one of our containers holding GPU memory (the sim), `baseline_mib` = GPU memory in use
    before the run (other sessions)."""
    if not samples:
        return {}
    ours = [{n: m for n, m in s["containers"].items() if n.startswith(prefixes)} for s in samples]
    slots = max(len(o) for o in ours)
    peak_ours = max(sum(o.values()) for o in ours)
    return {"samples": len(samples), "slots_seen": slots, "gpu_util_mean": round(mean(s["gpu_util"] for s in samples), 1),
            "gpu_util_max": max(s["gpu_util"] for s in samples), "gpu_mib_peak": max(s["gpu_mib"] for s in samples),
            "gpu_mib_over_baseline_peak": round(max(s["gpu_mib"] for s in samples) - baseline_mib),
            "gpu_mib_ours_peak": round(peak_ours), "gpu_mib_per_slot": round(peak_ours / slots) if slots else None,
            "cpu_mean": round(mean(s["cpu"] for s in samples), 1), "cpu_max": round(max(s["cpu"] for s in samples), 1),
            "ram_gb_peak": round(max(s["ram_gb"] for s in samples), 1), "load1_max": round(max(s["load1"] for s in samples), 1),
            "containers_peak": max(len(s["running"]) for s in samples)}


class ClientTimes:
    """Wall time of every client container run (the seam both runners share: ClientContainer.run) and the faults it raised."""

    def __init__(self, clock=time.monotonic):
        self.clock, self.runs, self.faults = clock, [], []
        self.lock = threading.Lock()

    def wrap(self, original):
        def run(client, log, *a, **kw):
            t0 = self.clock()
            try:
                rc = original(client, log, *a, **kw)
            except BaseException as e:
                with self.lock:
                    self.faults.append({"log": str(log), "fault": f"{type(e).__name__}: {e}"[:200]})
                raise
            finally:
                with self.lock:
                    self.runs.append({"log": str(log), "t0": t0, "wall_s": self.clock() - t0})
            return rc
        return run

    @contextmanager
    def patched(self, cls):
        original = cls.run
        cls.run = self.wrap(original)
        try:
            yield self
        finally:
            cls.run = original


def rtf(sim_s: float, wall_s: float) -> float | None:
    return round(sim_s / wall_s, 3) if wall_s > 0 else None


def rollout_rtfs(runs: list[dict], sim_s_of) -> list[dict]:
    """Per rollout client run (log `client.log` in the rollout dir): wall, sim s (`sim_s_of(log path)` or None), RTF."""
    rows = []
    for r in runs:
        if Path(r["log"]).name != "client.log" or (sim := sim_s_of(Path(r["log"]))) is None:
            continue
        rows.append({"wall_s": round(r["wall_s"], 1), "sim_s": round(sim, 1), "rtf": rtf(sim, r["wall_s"])})
    return rows


def birth_s(text: str) -> float:
    """`stat -c %w` ("2026-11-24 23:45:08.123456789 +0530") -> epoch seconds."""
    day, clock, zone = text.split()
    whole, _, frac = clock.partition(".")
    stamp = datetime.strptime(f"{day} {whole}.{(frac + '000000')[:6]} {zone}", "%Y-%m-%d %H:%M:%S.%f %z")
    return stamp.timestamp()


def bag_sim_s(metadata: str) -> float | None:
    found = re.search(r"duration:\s*nanoseconds:\s*(\d+)", metadata)
    return int(found.group(1)) / 1e9 if found else None


def _birth(path: Path) -> float:
    return birth_s(_run("stat", "-c", "%w", str(path)))


def bag_rtfs(root: Path, birth=_birth) -> list[dict]:
    """The sim's real-time factor per recorded episode under `root`: the bag's sim duration over the wall time between the creation of
    its data file (recording starts) and of metadata.yaml (written at close). Inside the episode: no client or reset overhead."""
    rows = []
    for meta in sorted(root.rglob("metadata.yaml")):
        data, sim = next(iter(sorted(meta.parent.glob("*.mcap")) + sorted(meta.parent.glob("*.db3"))), None), bag_sim_s(meta.read_text())
        if data is None or sim is None:
            continue
        wall = birth(meta) - birth(data)
        if wall > 0:
            rows.append({"bag": str(meta.parent), "sim_s": round(sim, 1), "wall_s": round(wall, 1), "rtf": rtf(sim, wall)})
    return rows


def stability(base: dict[str, set], other: dict[str, set]) -> dict:
    """Verdicts per config (a set of (passed, failed_phase)) against the N=1 run: configs seen in both, those whose verdict set differs."""
    common = sorted(set(base) & set(other))
    differ = [c for c in common if other[c] != base[c]]
    return {"configs": len(common), "same": len(common) - len(differ), "differ": differ}
