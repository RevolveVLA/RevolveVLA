"""The sidecar of a live episode: the flags a bag cannot carry (record topics are frozen). It is `episode_<n>.interventions.json`
next to the episode's `episode_<n>.json` in the sim's out (one sim holds several episodes, so each has its own file).

  {takeover_t, takeover_sim_t, trigger, reason, phase, max_delta_action, expert, expert_failure,
   segments: [[t0, t1, "policy" | "expert"], ...], segment: [t0, t1] | null}

With a hand-back (`live.handback`) the first intervention's data stays at the top level and the sidecar adds
  kept: [[t0, t1], ...] | null     (every expert span when `record: interventions`; `segment` is then the span iff there is one)
  interventions: [{trigger, reason, phase, takeover_t, handback_t | null, handback_reason | null,
                   max_delta_action: {takeover, handback}, expert_failure}, ...]
`segments` alternates policy / expert over the whole episode (several expert spans).

Times are sim seconds since the episode's start (`takeover_sim_t`: the sim clock). Without a takeover `takeover_t` and the
other takeover keys are null and `segments` is the policy's whole episode. `segment` is the span a recorder that keeps only
the intervention asks for (the manifest copies it). The recipe reads the same name and `segments` (its plugin sources/recovery)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Mapping

SUFFIX = ".interventions.json"


def sidecar_of(result: str | Path) -> Path:
    """The sidecar of the episode whose result file is `result` (`.../episode_3.json`)."""
    p = Path(result)
    return p.with_name(p.stem + SUFFIX)


def write(out: Path, episode: int, flags: Mapping) -> Path:
    path = sidecar_of(Path(out) / f"episode_{episode}.json")
    path.write_text(json.dumps(flags, indent=1))
    return path


def read(result: str | Path) -> dict | None:
    """The flags of the episode with this result file; None when it has none (or they are unreadable)."""
    try:
        return json.loads(sidecar_of(result).read_text())
    except (OSError, ValueError):
        return None


def tally(root: Path) -> Counter:
    """Takeovers per phase over every sidecar under `root` (each intervention of an episode counts)."""
    taken: Counter = Counter()
    for path in Path(root).rglob("episode_*" + SUFFIX):
        try:
            f = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        for phase in (i.get("phase") for i in f.get("interventions") or [f]):
            if phase:
                taken[phase] += 1
    return taken
