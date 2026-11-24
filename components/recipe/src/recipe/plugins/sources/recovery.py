"""source recovery: the episodes of a recovery run's `manifest.json` (datagen's entries + a `recovery` block). Params: `run` (the
recovery run's out), `kind` (phase | dagger; default both), `passed` (true default | false | any), `segment` (full default |
interventions). Item phase = the phase it recovers (phase kind) or where the takeover happened (dagger); the scene is the EVALUATION
scene it came from (`recovery.from_rollout.scene`), so it splits and is guarded like the rollouts of that scene.

`segment: interventions` keeps, of a dagger episode, only the expert's spans (one item each): the manifest's `recovery.segments`
[[t0, t1], ...] (a hand-back run) or `recovery.segment` [t0, t1] when the recorder set it,
else the `expert` entries of `segments` in the episode's sidecar (`<result stem>.interventions.json` next to the result file, e.g.
`episode_3.interventions.json`: {takeover_t, ..., segments: [[t0, t1, "policy" | "expert"], ...]}; one item per expert span).
Seconds from the episode's start.
A dagger episode with no such span is left out (`no_segment`). Phase-wise episodes are the expert's throughout: always whole."""
from __future__ import annotations

import os
from collections import Counter
from typing import Any, Mapping

from ...core.errors import RecipeError
from ...core.files import duration, read_json, resolve, scene_seed
from ...core.interfaces import Source
from ...core.items import Item, Read
from ...core.params import Params
from ...core.registry import REGISTRY

KINDS, SEGMENTS, SIDECAR = ("phase", "dagger"), ("full", "interventions"), ".interventions.json"


def expert_spans(entry: Mapping, result: str) -> list[tuple[float, float]]:
    """The expert's [t0, t1] spans of a dagger episode: from its manifest entry, else its sidecar (none when neither says)."""
    if segs := entry["recovery"].get("segments"):
        return [(float(a), float(b)) for a, b in segs]
    if seg := entry["recovery"].get("segment"):
        return [(float(seg[0]), float(seg[1]))]
    sidecar = os.path.splitext(result)[0] + SIDECAR
    if not os.path.isfile(sidecar):
        return []
    try:
        return [(float(a), float(b)) for a, b, who in read_json(sidecar, "an interventions file")["segments"] if who == "expert"]
    except (KeyError, TypeError, ValueError):
        raise RecipeError(f"{sidecar}: expected segments: [[t0, t1, 'policy' | 'expert'], ...]") from None


@REGISTRY.register("source", "recovery")
class RecoverySource(Source):
    def __init__(self, name: str, params: Mapping[str, Any], root):
        p = Params(f"sources.{name}", params, ("run", "kind", "passed", "segment"))
        self.name, self.root, self.run, self.want = name, root, p.path("run", root), p.passed()
        self.only, self.segment = p.choice("kind", KINDS, None), p.choice("segment", SEGMENTS, "full")
        self.kind = self.only or "recovery"

    def read(self) -> Read:
        data = read_json(os.path.join(self.run, "manifest.json"), "a recovery manifest")
        items, skipped = [], Counter()
        for e in data["episodes"]:
            rec = e["recovery"]
            result = resolve(self.run, e["result"])
            if self.only and rec["kind"] != self.only or self.want is not None and bool(e["passed"]) != self.want:
                skipped["not_wanted"] += 1
            elif not (e.get("bag") and os.path.isdir(resolve(self.run, e["bag"]))):
                skipped["no_bag"] += 1
            else:
                spans = expert_spans(e, result) if self.segment == "interventions" and rec["kind"] == "dagger" else [None]
                if not spans:
                    skipped["no_segment"] += 1
                scene = resolve(self.root, rec["from_rollout"]["scene"])
                labels = {"episode": e["episode"], "trigger": rec.get("trigger"), "reason": rec.get("reason"),
                          **{k: rec["from_rollout"].get(k) for k in ("set", "rollout", "try")}}
                items += [Item(rec["kind"], self.name, rec["phase"], scene, scene_seed(scene), os.path.dirname(result), result,
                               resolve(self.run, e["bag"]), span, bool(e["passed"]), e.get("task", data.get("task")),
                               e.get("robot", data.get("robot")), duration(result), labels) for span in spans]
        return Read(items, skipped)
