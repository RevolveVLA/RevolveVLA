"""The capacity bench's helpers (CPU): parsers, sampler, client timing, RTF, verdict stability."""
import subprocess

import pytest
from bench_metrics import (ClientTimes, Sampler, container_id, cpu_busy, cpu_percent, gpu_by_container, parse_apps, parse_gpu, ram_used_gb,
                           rollout_rtfs, rtf, stability, summarise)

CID = "ab" * 32


def test_the_nvidia_smi_csv_is_parsed():
    assert parse_gpu("37, 13132\n") == (37.0, 13132.0)
    assert parse_apps("4952, 432\n279415, 7064\n") == {4952: 432.0, 279415: 7064.0}
    assert parse_apps("") == {}


def test_cpu_percent_is_the_busy_share_between_two_reads():
    a = cpu_busy("cpu  100 0 100 700 100 0 0 0 0 0")                 # idle + iowait = 800 of 1000
    b = cpu_busy("cpu  200 0 200 1400 200 0 0 0 0 0")
    assert a == (200.0, 1000.0) and cpu_percent(a, b) == pytest.approx(100.0 * 200 / 1000)
    assert cpu_percent(a, a) == 0.0


def test_ram_used_is_total_minus_available_in_gb():
    assert ram_used_gb("MemTotal: 2097152 kB\nMemFree: 1 kB\nMemAvailable: 1048576 kB\n") == pytest.approx(1.0)


def test_the_container_id_is_read_from_both_cgroup_layouts():
    assert container_id(f"0::/system.slice/docker-{CID}.scope\n") == CID
    assert container_id(f"12:cpu:/docker/{CID}\n") == CID
    assert container_id("0::/user.slice/session-3.scope") is None


def test_gpu_memory_is_summed_per_container_and_host_processes_are_apart():
    apps = {1: 100.0, 2: 50.0, 3: 7000.0}
    cgroups = {1: f"0::/docker-{CID}.scope", 2: f"0::/docker-{CID}.scope", 3: "0::/user.slice"}
    assert gpu_by_container(apps, cgroups, {CID: "ev-1-sim"}) == {"ev-1-sim": 150.0, "host": 7000.0}


def test_the_sampler_reads_one_sample_through_its_seams():
    texts = {"/proc/stat": "cpu  100 0 100 700 100 0 0 0\n", "/proc/meminfo": "MemTotal: 2097152 kB\nMemAvailable: 1048576 kB\n",
             "/proc/9/cgroup": f"0::/docker-{CID}.scope\n"}
    out = {"--query-gpu=utilization.gpu,memory.used": "12, 1000\n", "--query-compute-apps=pid,used_memory": "9, 300\n",
           "ps": f"{CID} ev-abc-0\n"}

    def run(*argv):
        return next(v for k, v in out.items() if k in argv)
    sample, busy = Sampler(run=run, read=lambda p: texts.get(p, ""), clock=lambda: 5.0).sample((0.0, 0.0))
    assert sample["gpu_util"] == 12 and sample["gpu_mib"] == 1000 and sample["containers"] == {"ev-abc-0": 300.0}
    assert sample["running"] == ["ev-abc-0"] and sample["ram_gb"] == pytest.approx(1.0) and busy == (200.0, 1000.0)


def test_the_summary_counts_our_containers_as_slots():
    samples = [{"gpu_util": 10, "gpu_mib": 14000, "cpu": 20, "ram_gb": 30, "load1": 4, "containers": {"ev-a-sim": 800, "host": 7000}, "running": ["a"]},
               {"gpu_util": 60, "gpu_mib": 15000, "cpu": 40, "ram_gb": 32, "load1": 6,
                "containers": {"ev-a-sim": 900, "ev-b-sim": 1100, "host": 7000}, "running": ["a", "b", "c"]}]
    s = summarise(samples, ("ev-",), baseline_mib=13000)
    assert s["slots_seen"] == 2 and s["gpu_mib_ours_peak"] == 2000 and s["gpu_mib_per_slot"] == 1000
    assert s["gpu_mib_over_baseline_peak"] == 2000 and s["gpu_util_max"] == 60 and s["cpu_mean"] == 30 and s["containers_peak"] == 3
    assert summarise([], ("ev-",), 0) == {}


class Fake:
    def __init__(self, fail=None):
        self.fail = fail

    def run(self, log, *a, **kw):
        if self.fail:
            raise self.fail
        return 0


def test_client_runs_are_timed_and_their_faults_recorded_and_reraised():
    ticks = iter([1.0, 4.0, 10.0, 11.5])
    times = ClientTimes(clock=lambda: next(ticks))
    with times.patched(Fake):
        assert Fake().run("a/client.log") == 0
        with pytest.raises(subprocess.TimeoutExpired):
            Fake(subprocess.TimeoutExpired("x", 1)).run("b/client.log")
    assert [r["wall_s"] for r in times.runs] == [3.0, 1.5] and len(times.faults) == 1 and "TimeoutExpired" in times.faults[0]["fault"]
    assert Fake().run("c") == 0 and len(times.runs) == 2                            # unpatched again


def test_the_rtf_is_sim_seconds_per_wall_second_for_rollout_clients_only(tmp_path):
    runs = [{"log": str(tmp_path / "r0" / "client.log"), "wall_s": 40.0}, {"log": str(tmp_path / "slot.client.log"), "wall_s": 90.0},
            {"log": str(tmp_path / "r1" / "client.log"), "wall_s": 30.0}]
    sims = {str(tmp_path / "r0" / "client.log"): 20.0}
    rows = rollout_rtfs(runs, lambda p: sims.get(str(p)))                          # r1 has no episode.json; the job's log is not a rollout
    assert rows == [{"wall_s": 40.0, "sim_s": 20.0, "rtf": 0.5}] and rtf(1.0, 0.0) is None


def test_stability_compares_the_verdicts_per_config():
    base = {"1": {(True, None)}, "2": {(False, "open")}, "3": {(False, "approach")}}
    other = {"1": {(True, None)}, "2": {(True, None)}, "9": {(True, None)}}
    assert stability(base, other) == {"configs": 2, "same": 1, "differ": ["2"]}
