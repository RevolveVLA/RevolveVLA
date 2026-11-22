"""sample -> check -> rollout -> report, over injected plugins.

`sample` builds every eval set (configs + start dirs, resumable), `check` reports the sets' checks, `rollout` runs
`rollouts` policy runs per config and judges each, `report` writes report.md. Stages that need a built set build it
(idempotent), so each can run alone. Rollout records and verdicts are kept in `<out>/results.json`, saved after every
rollout, so a killed run resumes; a changed spec hash refuses an existing `out`. Layout: `<out>/<set>/<seed>_r<i>/`.

Retries (`spec.retries`) are the only second-attempt mechanism and belong to a later PRD (walk); the seam is the
Rollout's `restore` argument."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .errors import EvalError
from .interfaces import Built, CkptSource, EvalSet, Judge, Policy, ResolvedCkpt, Rollout
from .sets import repo_root
from .spec import EvalSpec
from .task import load_task

STAGES = ("sample", "check", "rollout", "report")


class PipelineError(EvalError):
    pass


@dataclass
class PluginSet:
    sets: Mapping[str, EvalSet] = field(default_factory=dict)      # by set name
    ckpt: CkptSource | None = None
    policy: Policy | None = None
    rollout: Rollout | None = None
    judge: Judge | None = None


class Pipeline:
    def __init__(self, spec: EvalSpec, plugins: PluginSet, out: Path, dry_run: bool = False,
                 log: Callable[[str], None] = print):
        self.spec, self.plugins, self.out, self.dry_run, self.log = spec, plugins, out, dry_run, log
        self.gui = not spec.run.headless
        self.workers = (spec.run.workers or 1) if spec.run.headless else 1     # a GUI run has one window
        self.results_path = out / "results.json"
        self.results = self._open()
        self._built: dict[str, Built] = {}

    def _open(self) -> dict:
        if not self.results_path.exists():
            return {"spec_hash": self.spec.hash, "ckpt": None, "sets": {}}
        data = json.loads(self.results_path.read_text())
        if data["spec_hash"] != self.spec.hash:
            raise PipelineError(f"{self.out} holds the results of another spec (hash {data['spec_hash']}, this one "
                                f"{self.spec.hash}): use another `out` or delete it")
        return data

    def run(self, stages: Sequence[str] = STAGES, sets: Sequence[str] | None = None) -> bool:
        """Run `stages` on `sets` (default all); False when a check failed."""
        if unknown := [n for n in sets or () if n not in self.spec.sets]:
            raise PipelineError(f"unknown set {unknown} (sets: {list(self.spec.sets)})")
        names = [n for n in self.spec.sets if sets is None or n in sets]
        if self.dry_run:
            self._plan(stages, names)
            return True
        self.out.mkdir(parents=True, exist_ok=True)
        ok = True
        for stage in stages:
            ok &= getattr(self, f"_{stage}")(names) is not False
        return ok

    def _plan(self, stages, names):
        s, say = self.spec, lambda m: self.log(f"[dry-run] {m}")
        say(f"stages: {', '.join(stages)}; out {self.out}")
        say(f"ckpt: {s.ckpt.source} {dict(s.ckpt.params)} (nothing resolved)" if s.ckpt else "ckpt: none")
        say(f"policy {s.policy.plugin}, rollout {s.rollout.plugin}, judge {s.judge.plugin}, "
            f"runner {s.runner.actions_per_chunk} actions per chunk, chunk_size_threshold {s.runner.chunk_size_threshold}")
        runs = 0
        for n in names:
            t = s.sets[n]
            pool = " + ".join(f"[{lo}, {hi}]" for lo, hi in t.pool.ranges()) if t.pool else ""
            md = f", min_distance {t.min_distance.value}{' realised' if t.min_distance.realised else ''}" if t.min_distance else ""
            what = (f"{t.count} configs from training split {t.source!r}" if t.kind == "train" else
                    f"{t.count} held-out configs, {t.sampler.strategy} sampler, pool {pool}{md}")
            say(f"set {n} ({t.plugin}): {what}")
            runs += t.count * s.rollouts
        say(f"{s.rollouts} rollouts per config, retries {s.retries}: {runs} first tries, "
            f"up to {runs * (1 + s.retries)} runs")
        workers = self.workers if s.run.workers or self.gui else "by free GPU memory"
        say(f"workers: {workers}, {'headless' if s.run.headless else 'gui'}")

    # ---------------------------------------------------------------- stages
    def _build(self, name: str) -> Built:
        if name not in self._built:
            self._built[name] = self.plugins.sets[name].build(self.spec.sets[name], self.out, self.workers)
            self.log(f"sample {name}: {len(self._built[name].items)} configs")
        return self._built[name]

    def _sample(self, names):
        for n in names:
            self._build(n)

    def _check(self, names) -> bool:
        ok = True
        for n in names:
            for check, r in self._build(n).checks.items():
                self.log(f"check {n} {check}: {'PASS' if r.ok else 'FAIL'}" + (f" ({r.reason})" if r.reason else ""))
                ok &= r.ok
            self.results["sets"].setdefault(n, {})["checks"] = {c: r.to_dict() for c, r in self._build(n).checks.items()}
        self._save()
        return ok

    def _rollout(self, names):
        p = self.plugins
        if p.policy.needs_ckpt and self.spec.ckpt is None:
            raise PipelineError(f"policy {self.spec.policy.plugin!r} needs a `ckpt:`")
        ckpt = p.ckpt.resolve(self.out) if p.ckpt and p.policy.needs_ckpt else None
        self.results["ckpt"] = ckpt and {"path": ckpt.path, "origin": ckpt.origin, "changes": ckpt.changes}
        task = load_task(self.spec.train.task, repo_root(self.spec, self.out))
        for n in names:
            done = self.results["sets"].setdefault(n, {}).setdefault("rollouts", {})
            for item in self._build(n).items:
                for i in range(self.spec.rollouts):
                    key = f"{item.config.seed}_r{i}"
                    if key in done:
                        continue
                    record = p.rollout.run(item, self.out / n / key, seed=item.config.seed, runner=self.spec.runner,
                                           policy=p.policy, ckpt=ckpt, gui=self.gui)
                    v = p.judge.score(record, self.out / n / key, task)
                    done[key] = {"seed": item.config.seed, "record": record, "passed": v.passed, "detail": dict(v.detail)}
                    self._save()
                    self.log(f"rollout {n} {key}: {'PASS' if v.passed else 'FAIL'}")

    def _report(self, names):
        lines = [f"# Evaluation {self.spec.hash}", ""]
        for n in names:
            rolls = self.results["sets"].get(n, {}).get("rollouts", {})
            passed = sum(r["passed"] for r in rolls.values())
            lines.append(f"- {n}: {passed}/{len(rolls)} rollouts passed ({self.spec.judge.plugin})")
        (self.out / "report.md").write_text("\n".join(lines) + "\n")

    def _save(self):
        tmp = self.results_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.results, indent=1))
        os.replace(tmp, self.results_path)
