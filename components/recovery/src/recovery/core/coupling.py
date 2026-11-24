"""The recovery coupled to an evaluation run: `recovery: {spec: specs/recovery/<x>.yaml}` in the eval spec (evaluation.cli calls `attach`).

The recovery spec may have both `phases` and `live` here: the evaluation provides the run, so `from.eval` is not read (it is the
evaluation's own out) and the standalone split (a spec for a live run, a run dir for phases) does not apply.
- `live`: the evaluation's policy runs under the recorder (core/live.py `wrap`) in the sets `from.sets` (default: the ones that are not
  held-out; the leakage guard of the standalone run applies), found by the seed of the config; the other sets run the plain policy. Its sidecars sit next to the
  evaluation's episode results.
- `phases`: when a config's rollouts are committed the failures of their tries are scanned (the spec's scanner, only those rollouts)
  and the quota (core/quota.py `Admission`) picks the jobs, which the evaluation's worker slots run (evaluation core/slots.py) when no
  rollout is pending: the same core as `./revolve recover` (`Pipeline.work`, `record`, the same state, manifest and report) into
  `<eval out>/recovery/`. A job takes a slot's sim of its own (the runner's slots, warm-reset between jobs under `run.reset: warm`);
  the slot's eval sim is closed first, so there are never more sims than worker slots.
`run.workers`, `reset` and `headless` are the evaluation's."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

from datagen.core.spec import SpecError
from evaluation.core.interfaces import Follow
from revolve_common import specs

from . import leakage
from . import live as lv
from .pipeline import Pipeline, PluginSet, phase_plugins
from .registry import REGISTRY
from .robot import arm_of
from .scan import checked
from .spec import RecoverySpec, RunSpec, load

lv.register_policies()          # the test policies are plugins of the evaluation registry: before the spec's policy is built


class Coupled(Follow):
    def __init__(self, pipeline: Pipeline, eval_out: Path, log: Callable[[str], None] = print):
        self.pipeline, self.eval_out, self.log = pipeline, eval_out, log
        self.queued: set[str] = set()                           # job ids handed to the slots (this process)

    @property
    def spec(self) -> RecoverySpec:
        return self.pipeline.spec

    # ---- Follow
    def after(self, set_name: str, entries: Mapping[str, Mapping]) -> Sequence[Callable[[], None]]:
        if self.spec.phases is None:
            return []
        p = self.pipeline
        scanned = set(p.state.data.get("scanned", ()))
        fresh = sorted(k for k in entries if f"{set_name}/{k}" not in scanned)
        if fresh:
            scan = p.plugins.scanner.scan(p.out, {(set_name, k) for k in fresh})
            p.absorb(scan)
            with p.lock:
                p.state.data["scanned"] = sorted(scanned | {f"{set_name}/{k}" for k in fresh})
                p.state.save()
            self.log(f"recover {set_name}: {len(scan.failures)} failures in {len(fresh)} rollouts")
        jobs = [(i, f) for i, f in p.todo() if f.id not in self.queued]
        self.queued.update(f.id for _, f in jobs)
        return [lambda i=i, f=f: p.record(f, p.work(i, f)) for i, f in jobs]

    def flush(self) -> None:
        if self.pipeline.plugins.runner:
            self.pipeline.plugins.runner.close()
        self.pipeline.out.mkdir(parents=True, exist_ok=True)
        self.pipeline.report()
        if self.spec.live:
            line = lv.summary(self.eval_out)
            (self.pipeline.out / "live_summary.json").write_text(json.dumps(line, indent=1))
            self.log("live " + json.dumps(line))

    def check(self) -> bool:
        return self.spec.phases is None or self.pipeline.run(("check",))

    def report(self) -> str:
        s, lines = self.spec, ["## Recovery", ""]
        if s.live:
            v = lv.summary(self.eval_out)
            lines.append(f"- {s.live.plugin}: the policy alone passed {v['policy_only_success']} of {v['episodes']} episodes in the "
                         f"live sets; {v['takeovers']} taken over ({', '.join(f'{k} {n}' for k, n in sorted(v['reasons'].items())) or 'none'})"
                         f", {v['recovered']} of them passed with the expert (the verdicts above include them).")
        if s.phases:
            jobs = self.pipeline.state.data.get("jobs", {})
            lines.append(f"- phase-wise: {len(self.pipeline.state.data.get('selection', []))} jobs selected, "
                         f"{sum(j['status'] == 'done' for j in jobs.values())} done, {sum(j['status'] != 'done' for j in jobs.values())} "
                         f"short or failed: recovery/report.md")
        return "\n".join(lines) + "\n"

    def stop(self) -> None:
        if self.pipeline.plugins.runner:
            self.pipeline.plugins.runner.stop()


def coupled_spec(spec: RecoverySpec, eval_spec, out: Path) -> RecoverySpec:
    """`spec` for the evaluation `eval_spec` that runs into `out`: the run it reads is that out, it writes `<out>/recovery` and
    executes as the evaluation does (workers, reset, headless)."""
    if (spec.task, spec.robot) != (eval_spec.train.task, eval_spec.train.robot):
        raise SpecError(f"recovery: the spec is for {spec.task} / {spec.robot}, the evaluation for "
                        f"{eval_spec.train.task} / {eval_spec.train.robot}")
    run = RunSpec((eval_spec.run.workers or 1) if eval_spec.run.headless else 1, eval_spec.run.reset, eval_spec.run.headless)
    return dataclasses.replace(spec, eval_run=str(out), out=str(out / "recovery"), run=run)


def attach(eval_spec, plugins, root: Path, out: Path, log: Callable[[str], None] = print) -> None:
    """Couple the recovery spec of `eval_spec.recovery` to `plugins` (the evaluation's PluginSet, to run into `out`): its policy
    goes under the live recorder and `plugins.follow` is the phase-wise recovery. SpecError when the recovery spec does not fit."""
    path = Path(eval_spec.recovery.spec)
    spec = checked(coupled_spec(load(path if path.is_absolute() else root / path), eval_spec, out))
    if spec.live:
        run, reserved = leakage.sets_of(eval_spec, spec.eval_sets, spec.heldout)
        leakage.check(eval_spec, run, reserved)
        seeds = [r for n in run for r in leakage.pool_of(eval_spec, n)]
        lv.wrap(plugins, spec, REGISTRY.create("live", spec.live.plugin, spec, spec.live.params), out, seeds)
    pipeline = Pipeline(spec, arm_of(spec.robot, specs.REPO_SPECS), phase_plugins(spec, ("collect", "run")) if spec.phases
                        else PluginSet(), Path(spec.out), live_dir=out, log=log)
    plugins.follow = Coupled(pipeline, out, log)
