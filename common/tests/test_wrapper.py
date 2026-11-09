import json
import socket
import subprocess
import sys
import time

import httpx
import pytest

from revolve_common import job as J


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


DUMMY = r"""
import json, sys, time
n = int(sys.argv[1])
for i in range(1, n + 1):
    print("PROGRESS " + json.dumps({"step": i, "loss": round(1.0 / i, 4)}), flush=True)
    print("noise line", i, flush=True)
    time.sleep(0.1)
sys.exit(int(sys.argv[2]))
"""


def start(tmp_path, steps, code, linger=3.0):
    jd = tmp_path / "job"
    script = tmp_path / "dummy.py"
    script.write_text(DUMMY)
    J.write_model(jd / J.JOB, J.Job(id="t1", component="test",
                                    command=[sys.executable, "-u", str(script), str(steps), str(code)]))
    port = free_port()
    p = subprocess.Popen([sys.executable, "-m", "revolve_common.wrapper", "--job-dir", str(jd), "--host", "127.0.0.1",
                          "--port", str(port), "--linger-s", str(linger), "--grace-s", "2"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    url = f"http://127.0.0.1:{port}"
    for _ in range(200):
        try:
            httpx.get(url + "/status", timeout=1)
            break
        except httpx.HTTPError:
            time.sleep(0.05)
    else:
        p.kill()
        pytest.fail(f"wrapper did not come up: {p.stderr.read().decode()}")
    return p, url, jd


def test_progress_result_and_files(tmp_path):
    p, url, jd = start(tmp_path, steps=10, code=0)
    try:
        seen = []
        while True:
            st = httpx.get(url + "/status").json()
            seen.append(st["progress"].get("step"))
            if st["state"] not in ("pending", "running"):
                break
            assert httpx.get(url + "/result").status_code == 409
            time.sleep(0.15)
        mid = [s for s in seen if s is not None and s < 10]
        assert mid, f"no intermediate progress seen: {seen}"
        res = httpx.get(url + "/result").json()
        assert res["state"] == "succeeded" and res["returncode"] == 0
        assert res["progress"] == {"step": 10, "loss": 0.1}
        assert p.wait(timeout=20) == 0
    finally:
        p.kill()
    assert J.read_result(jd).state == "succeeded"
    st = J.read_status(jd)
    assert st.state == "succeeded" and st.progress["step"] == 10
    assert (jd / J.LOG).read_text().count("noise line") == 10
    assert not list(jd.glob(".*.tmp"))


def test_failure_exit_code(tmp_path):
    p, url, jd = start(tmp_path, steps=2, code=3, linger=0.5)
    assert p.wait(timeout=20) == 3
    r = J.read_result(jd)
    assert r.state == "failed" and r.returncode == 3


def test_cancel(tmp_path):
    p, url, jd = start(tmp_path, steps=1000, code=0)
    try:
        for _ in range(100):
            if httpx.get(url + "/status").json()["progress"].get("step", 0) >= 2:
                break
            time.sleep(0.05)
        assert httpx.post(url + "/cancel").json() == {"cancelled": True}
        for _ in range(100):
            if httpx.get(url + "/status").json()["state"] == "cancelled":
                break
            time.sleep(0.05)
        res = httpx.get(url + "/result").json()
        assert res["state"] == "cancelled" and res["returncode"] == -15
        assert p.wait(timeout=20) == 128 + 15
    finally:
        p.kill()
    assert J.read_result(jd).state == "cancelled"
    assert json.loads((jd / J.STATUS).read_text())["progress"]["step"] < 1000


def test_custom_parser():
    from revolve_common.wrapper import json_progress, load_parser
    assert load_parser(None) is json_progress
    assert json_progress('PROGRESS {"episodes": 3}') == {"episodes": 3}
    assert json_progress("PROGRESS not json") is None and json_progress("step 3") is None
    assert load_parser("revolve_common.wrapper:json_progress") is json_progress


def test_atomic_write(tmp_path):
    J.write_json(tmp_path / "a.json", {"x": 1})
    assert json.loads((tmp_path / "a.json").read_text()) == {"x": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]
