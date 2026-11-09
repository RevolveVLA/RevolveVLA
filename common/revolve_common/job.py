"""Job contract between the manager and a component.

  <job dir>/job.json      in:  what to run (written by the manager)
  <job dir>/status.json   out: live state + progress (rewritten by the wrapper while it runs)
  <job dir>/result.json   out: final state + return code (written once, at the end)
  <job dir>/stdout.log    out: the command's merged stdout/stderr

Every write is atomic (temp file in the same dir + os.replace), so a reader (or a manager restarted
mid-run) never sees a partial file.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

State = Literal["pending", "running", "succeeded", "failed", "cancelled"]
TERMINAL = ("succeeded", "failed", "cancelled")
JOB, STATUS, RESULT, LOG = "job.json", "status.json", "result.json", "stdout.log"


class Job(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    component: str
    command: list[str] = Field(min_length=1)
    params: dict[str, Any] = {}
    env: dict[str, str] = {}


class Status(BaseModel):
    job_id: str
    state: State = "pending"
    pid: int | None = None
    started_at: float | None = None
    updated_at: float = Field(default_factory=time.time)
    progress: dict[str, Any] = {}
    last_line: str | None = None


class Result(BaseModel):
    job_id: str
    state: State
    returncode: int | None
    started_at: float | None
    finished_at: float
    progress: dict[str, Any] = {}
    outputs: dict[str, Any] = {}
    error: str | None = None


def write_json(path, data):
    """Atomic JSON write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=1, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_model(path, model: BaseModel):
    write_json(path, model.model_dump(mode="json"))


def read_job(job_dir) -> Job:
    return Job.model_validate_json((Path(job_dir) / JOB).read_text())


def read_status(job_dir) -> Status | None:
    p = Path(job_dir) / STATUS
    return Status.model_validate_json(p.read_text()) if p.exists() else None


def read_result(job_dir) -> Result | None:
    p = Path(job_dir) / RESULT
    return Result.model_validate_json(p.read_text()) if p.exists() else None
