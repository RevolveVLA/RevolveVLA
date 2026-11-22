"""sample -> check -> rollout -> report, over injected plugins.

`sample` builds every eval set (configs + start dirs, resumable), `check` reports the sets' checks, `rollout` walks each
set's configs x `rollouts` (core/walk.py: parallel, retries, resume) and judges every try, `report` writes report.md.
Stages that need a built set build it (idempotent), so each can run alone. Everything is kept in `<out>/results.json`
(core/results.py), saved after every rollout, so a killed run resumes; a changed spec hash refuses an existing `out`.
Layout: `<out>/<set>/<seed>_r<i>/` (+ `retry_<j>/`). `abort` stops what the sample and rollout stages started (docker jobs)
when the run is interrupted."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .errors import EvalError
from .interfaces import Built, CkptSource, EvalSet, Judge, Policy, ResolvedCkpt, Rollout
from .results import Results
from .runner import check_chunk_size
from .sets import repo_root
from .spec import EvalSpec
from .task import load_task
from .termination import rollout_policy
from .walk import RetryPolicy, Walk

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
                 log: Callable[[str], None] = print, abort: Callable[[], None] | None = None):
        self.spec, self.plugins, self.out, self.dry_run, self.log, self.abort = spec, plugins, out, dry_run, log, abort
        self.gui = not spec.run.headless
        self.workers = (spec.run.workers or 1) if spec.run.headless else 1     # sample stage; a GUI run has one window
        self.results = Results.open(out / "results.json", spec.hash)
        self._built: dict[str, Built] = {}

    def rollout_workers(self) -> int:
        """Rollouts at a time: the GUI's one, `run.workers`, else the rollout plugin's default (free GPU memory)."""
        if self.gui:
            return 1
        return self.spec.run.workers or self.plugins.rollout.default_workers(self.plugins.policy)

    def run(self, stages: Sequence[str] = STAGES, sets: Sequence[str] | None = None) -> bool:
        """Run `stages` on `sets` (default all); False when a check failed."""
        if unknown := [n for n in sets or () if n not in self.spec.sets]:
            raise PipelineError(f"unknown set {unknown} (sets: {list(self.spec.sets)})")
        names = [n for n in self.spec.sets if sets is None or n in sets]
        if self.dry_run:
            self._plan(stages, names)
            return True
        self.out.mkdir(parents=True, exist_ok=True)
        self.results.data["training"] = {"plugin": self.spec.training.plugin, **self.spec.training.params}
        ok = True
        for stage in stages:
            ok &= getattr(self, f"_{stage}")(names) is not False
        return ok

    def _plan(self, stages, names):
        s, say = self.spec, lambda m: self.log(f"[dry-run] {m}")
        say(f"stages: {', '.join(stages)}; out {self.out}")
        say(f"ckpt: {s.ckpt.source} {dict(s.ckpt.params)} (nothing resolved)" if s.ckpt else "ckpt: none")
        say(f"policy {s.policy.plugin}, rollout {s.rollout.plugin}, judge {s.judge.plugin}, "
            f"runner {s.runner.describe()}")
        if s.runner.warning:
            say(f"warning: {s.runner.warning}")
        runs = 0
        for n in names:
            t = s.sets[n]
            pool = " + ".join(f"[{lo}, {hi}]" for lo, hi in t.pool.ranges()) if t.pool else ""
            md = f", min_distance {t.min_distance.value}" if t.min_distance else ""
            what = (f"{t.count} configs from training split {t.source!r}" if t.kind == "train" else
                    f"{t.count} held-out configs, {t.sampler.strategy} sampler, pool {pool}{md}")
            say(f"set {n} ({t.plugin}): {what}")
            runs += t.count * s.rollouts
        say(f"{s.rollouts} rollouts per config, retries {s.retries}: {runs} first tries, "
            f"up to {runs * (1 + s.retries)} runs")
        workers = self.workers if s.run.workers or self.gui else "by free GPU memory"
        say(f"workers: {workers}, {'headless' if s.run.headless else 'gui'}")
        pol = rollout_policy(None, s.rollout.params, None)          # the limits are the yaml's: the task has no `rollout:`
        say("rollout limits: " + ", ".join(f"{k} {v}" for k, v in pol.items() if k not in ("rtf", "wall_limit_s")) +
            (f", rtf {pol['rtf']}, wall limit {pol['wall_limit_s']} s" if pol["rtf"] else
             ", rtf measured at run time (runs/eval_rtf.json), wall limit = sim time / rtf x wall_safety"))

    # ---------------------------------------------------------------- stages
    def _build(self, name: str) -> Built:
        if name not in self._built:
            self._built[name] = self.plugins.sets[name].build(self.spec.sets[name], self.out, self.workers, self.abort)
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
            self.results.set(n)["checks"] = {c: r.to_dict() for c, r in self._build(n).checks.items()}
        self.results.save()
        return ok

    def _rollout(self, names):
        p, s = self.plugins, self.spec
        if p.policy.needs_ckpt and s.ckpt is None:
            raise PipelineError(f"policy {s.policy.plugin!r} needs a `ckpt:`")
        ckpt = p.ckpt.resolve(self.out) if p.ckpt and p.policy.needs_ckpt else None
        if s.runner.warning:
            self.log(f"warning: {s.runner.warning}")
        chunk_size = check_chunk_size(s.runner, ckpt)
        r = s.runner
        self.results.data["runner"] = {"n_action_steps": r.n_action_steps, "actions_per_chunk": r.actions_per_chunk,
                                       "chunk_size_threshold": r.chunk_size_threshold, "aggregate": r.aggregate,
                                       "ckpt_chunk_size": chunk_size, "text": r.describe()}
        self.results.data["ckpt"] = ckpt and {"path": ckpt.path, "origin": ckpt.origin, "changes": ckpt.changes}
        task = load_task(s.train.task, repo_root(s, self.out))
        policy = rollout_policy(None, s.rollout.params, p.rollout.rtf(p.policy))
        walk = Walk(p.rollout, p.policy, p.judge, RetryPolicy(s.retries), s.runner, ckpt, task, self.out,
                    self.rollout_workers(), self.gui, self.log, self.abort)
        for n in names:
            built = self._build(n)
            self.results.set(n)["rollout_policy"] = policy
            self.results.set(n)["configs"] = [dict(c.config.to_dict(), scene=str(c.scene)) for c in built.items]
            walk.run(n, built, s.rollouts, self.results.rollouts(n), lambda k, e, n=n: self.results.commit(n, k, e))

    def _report(self, names):
        kinds = {n: "seen" if t.kind == "train" else "new" for n, t in self.spec.sets.items()}
        (self.out / "report.md").write_text(self.results.report(self.spec.judge.plugin, names, kinds))
