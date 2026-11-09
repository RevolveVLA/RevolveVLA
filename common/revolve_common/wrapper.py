"""HTTP entrypoint wrapper: runs a component's real tool as a subprocess and reports on it.

  python -m revolve_common.wrapper --job-dir DIR [--port 8000] [--parser pkg.mod:fn] [-- CMD ...]

CMD defaults to job.json's command. The wrapper tees the command's output to stdout.log, feeds every
line to the progress parser (fn(line) -> dict or None, merged into status.progress), keeps
status.json current, writes result.json at the end and exits with the command's code (128 + signal
when it was killed). HTTP: GET /status, GET /result (409 until finished), POST /cancel (SIGTERM to
the command's process group, SIGKILL after --grace-s). The server stays up --linger-s after the end
so a poller can read /result; a manager that missed it reads result.json.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException

from . import job as J

PREFIX = "PROGRESS "


def json_progress(line):
    """Default parser: a line `PROGRESS {"step": 10, "loss": 0.3}` -> that dict."""
    if line.startswith(PREFIX):
        try:
            d = json.loads(line[len(PREFIX):])
            return d if isinstance(d, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def load_parser(spec):
    if not spec:
        return json_progress
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn)


class Runner:
    def __init__(self, job_dir, command, job_id, parser=json_progress, env=None, grace_s=10.0, write_every_s=1.0):
        self.dir, self.cmd, self.parser = Path(job_dir), list(command), parser
        self.env, self.grace_s, self.write_every_s = env or {}, grace_s, write_every_s
        self.lock = threading.Lock()
        self.status = J.Status(job_id=job_id)
        self.result: J.Result | None = None
        self.proc = None
        self.cancelled = threading.Event()
        self._last_write = 0.0

    # -------------------------------------------------- state files
    def _write_status(self, force=False):
        now = time.time()
        if force or now - self._last_write >= self.write_every_s:
            self.status.updated_at = now
            J.write_model(self.dir / J.STATUS, self.status)
            self._last_write = now

    def snapshot(self):
        with self.lock:
            return self.status.model_dump(mode="json")

    # -------------------------------------------------- control
    def cancel(self):
        with self.lock:
            if self.result is not None or self.proc is None:
                return False
            self.cancelled.set()
            pgid = self.proc.pid
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return True
        threading.Timer(self.grace_s, self._kill, args=(pgid,)).start()
        return True

    def _kill(self, pgid):
        if self.result is None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def run(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        log = open(self.dir / J.LOG, "a", buffering=1)
        error = None
        try:
            self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                         bufsize=1, start_new_session=True, env={**os.environ, **self.env})
        except OSError as e:
            rc, error = 127, f"could not start {self.cmd[0]}: {e}"
        else:
            with self.lock:
                self.status.state, self.status.pid, self.status.started_at = "running", self.proc.pid, time.time()
                self._write_status(force=True)
            for line in self.proc.stdout:
                log.write(line)
                sys.stdout.write(line)
                line = line.rstrip("\n")
                try:
                    upd = self.parser(line)
                except Exception as e:          # a parser bug must not kill the job
                    upd, error = None, f"progress parser: {e!r}"
                with self.lock:
                    self.status.last_line = line[-500:]
                    if upd:
                        self.status.progress.update(upd)
                        self._write_status(force=True)
                    else:
                        self._write_status()
            rc = self.proc.wait()
        log.close()
        state = "cancelled" if self.cancelled.is_set() else ("succeeded" if rc == 0 else "failed")
        with self.lock:
            self.status.state = state
            self._write_status(force=True)
            self.result = J.Result(job_id=self.status.job_id, state=state, returncode=rc,
                                   started_at=self.status.started_at, finished_at=time.time(),
                                   progress=self.status.progress, error=error)
            J.write_model(self.dir / J.RESULT, self.result)
        return 128 - rc if rc < 0 else rc


def make_app(runner: Runner):
    app = FastAPI(title="revolve component wrapper")

    @app.get("/status")
    def status():
        return runner.snapshot()

    @app.get("/result")
    def result():
        if runner.result is None:
            raise HTTPException(409, "not finished")
        return runner.result.model_dump(mode="json")

    @app.post("/cancel")
    def cancel():
        return {"cancelled": runner.cancel()}

    return app


def serve(runner, host, port, linger_s):
    server = uvicorn.Server(uvicorn.Config(make_app(runner), host=host, port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)   # not the main thread: uvicorn skips its signal handlers
    th.start()
    while not server.started and th.is_alive():
        time.sleep(0.02)
    if not th.is_alive():
        raise SystemExit(f"wrapper: HTTP server did not start on {host}:{port}")
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: runner.cancel())
    code = runner.run()
    time.sleep(linger_s)
    server.should_exit = True
    th.join(timeout=10)
    return code


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--parser", default=None, help="pkg.module:function (default: PROGRESS <json> lines)")
    ap.add_argument("--grace-s", type=float, default=10.0)
    ap.add_argument("--linger-s", type=float, default=5.0)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
    jd = Path(a.job_dir)
    env = {}
    if (jd / J.JOB).exists():
        jb = J.read_job(jd)
        job_id, cmd, env = jb.id, cmd or jb.command, jb.env
    else:
        job_id = jd.name
    if not cmd:
        ap.error("no command (give one after -- or in job.json)")
    runner = Runner(jd, cmd, job_id, load_parser(a.parser), env=env, grace_s=a.grace_s)
    sys.exit(serve(runner, a.host, a.port, a.linger_s))


if __name__ == "__main__":
    main()
