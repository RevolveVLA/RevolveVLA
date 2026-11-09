"""Wrapper progress parser for training jobs: lerobot-train's own output (local, over ssh, or streamed back from
Modal) plus the entry's `PROGRESS {json}` lines.

  lerobot-train log line (every log_freq steps; LeRobot 0.6.1 MetricsTracker, the line legacy
  train_supervisor.py parsed):  step:1K smpl:2K ep:4 epch:2.21 loss:0.512 grdn:3.200 lr:1.0e-04 updt_s:0.500
                                data_s:0.010 smp/s:4 mem_gb:5.12
  lerobot-train tqdm bar (stderr; \r becomes a line break in the wrapper's text pipe):
                                Training:  15%|##        | 3/20 [00:05<00:30,  1.52s/step]

-> status.progress keys: step, total_steps, loss, lr, grad_norm, s_per_step, eta_s, gpu_mem_gb, losses
([[step, loss], ...], the last 500) and whatever the entry reports (phase, init, episodes, ...). The step
of a log line is counted (n-th line x log_freq) when the entry reported log_freq, since LeRobot rounds it
("1K"). Module state: one parser per process (the wrapper runs one job).
"""
from __future__ import annotations

import re

from revolve_common.wrapper import json_progress

LOG = re.compile(r"step:(\S+) smpl:\S+ ep:\S+ epch:\S+ loss:(\S+) grdn:(\S+) lr:(\S+) updt_s:(\S+) data_s:(\S+)"
                 r"(?: smp/s:\S+)?(?: mem_gb:(\S+))?")
BAR = re.compile(r"Training:\s+\d+%\|.*\|\s*(\d+)/(\d+) \[([\d:]+)<([\d:?]+)")
SUFFIX = {"": 1, "K": 1e3, "M": 1e6, "B": 1e9}
KEEP = 500
_state = {"log_freq": None, "n": 0, "losses": []}


def reset():
    _state.update(log_freq=None, n=0, losses=[])


def big(s: str) -> int:
    """LeRobot's format_big_number back to a number ("1K" -> 1000)."""
    m = re.fullmatch(r"(-?[\d.]+)([KMB]?)", s)
    return int(float(m.group(1)) * SUFFIX[m.group(2)]) if m else -1


def seconds(s: str) -> int | None:
    if "?" in s:
        return None
    out = 0
    for part in s.split(":"):
        out = out * 60 + int(part)
    return out


def parse(line: str):
    d = json_progress(line)
    if d is not None:
        if "log_freq" in d:
            _state["log_freq"] = int(d["log_freq"])
        return d
    m = LOG.search(line)
    if m:
        _state["n"] += 1
        step = _state["n"] * _state["log_freq"] if _state["log_freq"] else big(m.group(1))
        loss = float(m.group(2))
        _state["losses"] = (_state["losses"] + [[step, loss]])[-KEEP:]
        out = dict(step=step, loss=loss, grad_norm=float(m.group(3)), lr=float(m.group(4)),
                   s_per_step=round(float(m.group(5)) + float(m.group(6)), 4), losses=list(_state["losses"]))
        if m.group(7):
            out["gpu_mem_gb"] = float(m.group(7))
        return out
    m = BAR.search(line)
    if m:
        out = dict(step=int(m.group(1)), total_steps=int(m.group(2)), elapsed_s=seconds(m.group(3)))
        eta = seconds(m.group(4))
        if eta is not None:
            out["eta_s"] = eta
        return out
    return None


_serve = {"chunks": 0}


def parse_serve(line: str):
    """Serve jobs: ready once Rosetta's server bound its port (the line legacy evaluate.py waited for), then a
    count of the action chunks it generated."""
    if "RosettaPolicyServer started on" in line:
        return dict(ready=True, address=line.rsplit(" ", 1)[-1])
    if "Action chunk #" in line and " generated" in line:
        _serve["chunks"] += 1
        return dict(chunks=_serve["chunks"])
    return json_progress(line)
