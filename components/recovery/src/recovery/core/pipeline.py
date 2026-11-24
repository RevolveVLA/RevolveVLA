"""collect -> run -> check over injected plugins; `report.md` + `manifest.json` after every run.

`collect` reads the failures of the spec's eval run (the scanner: bags are read where rosbag2_py is) and keeps them in
state.json, so a resumed run works on the same failures. `run` picks the failures to recover (core/quota.py; also kept),
makes a job of each (core/jobs.py: its scene instance at the restart state), has the generator plan it (`Generator.plan`: a
planner that works offline per scene instance) and runs them on `spec.run.workers` slots
(datagen's pool.ordered: the jobs run in parallel, their results are committed in order), each recording the episodes of the
spec's datagen generator. `check` judges the run: every job ran and every phase has its quota of episodes. A killed run resumes:
collected failures, the selection and finished jobs are kept, a job continues from its own log (datagen.json).
Layout under `out`: state.json, manifest.json, report.md, <phase>/<failure id>/{scene.yaml, datagen.json, resets.json},
sims/<slot>/ (the sims' episode results and bags), logs/."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from datagen.core import episodes, pool
from datagen.core.interfaces import Generator
from datagen.core.state import State

from . import jobs as jb
from . import manifest as mf
from . import quota, report
from .errors import JobError, RecoveryError
from .failures import Failure
from .interfaces import LiveRecorder, Runner, Scanner
from .robot import Arm
from .spec import RecoverySpec

STAGES = ("collect", "run", "check")


class PipelineError(RecoveryError):
    pass


@dataclass
class PluginSet:
    scanner: Scanner | None = None
    runner: Runner | None = None
    generator: Generator | None = None
    live: LiveRecorder | None = None


class Pipeline:
    def __init__(self, spec: RecoverySpec, arm: Arm, plugins: PluginSet, out: Path, dry_run: bool = False,
                 log: Callable[[str], None] = print):
        self.spec, self.arm, self.plugins, self.out, self.dry_run, self.log = spec, arm, plugins, out, dry_run, log
        self.state = State.open(out / "state.json", spec, persist=not dry_run)
        self.workers = spec.run.workers if spec.run.headless else 1
        if self.workers != spec.run.workers:
            log(f"run.headless false: one GUI window, so workers {spec.run.workers} -> 1")

    def run(self, stages: Sequence[str] = STAGES) -> bool:
        """Run `stages`; False when a check failed."""
        if self.dry_run:
            self._plan(stages)
            return True
        self.out.mkdir(parents=True, exist_ok=True)
        ok = True
        try:
            for stage in stages:
                ok &= getattr(self, f"_{stage}")() is not False
        finally:
            self._report()
        return ok

    def _plan(self, stages):
        s, say = self.spec, lambda m: self.log(f"[dry-run] {m}")
        p = s.phases
        say(f"stages: {', '.join(stages)}; out {self.out}")
        say(f"collect: failures of the eval run {s.eval_run}, triggers {', '.join(t['plugin'] for t in s.triggers)}")
        say(f"restart: {p.restart_from}" + (f", {p.back_s} s before the failure" if p.restart_from == "before_failure" else "")
            + f"; unreproducible when a scene joint is beyond {p.max_object_joint}")
        say("quota (episodes): " + (", ".join(f"{k} {n}" for k, n in p.quota.items()) or "none")
            + f"; {p.per_failure} per failure, up to {p.episodes.attempts} tries")
        say(f"generator {p.generator.plugin}, runner {s.runner.plugin}, keep {s.keep}"
            + (f", live {s.live.plugin}" if s.live else ""))
        say(f"workers: {self.workers}, reset {s.run.reset}, {'headless' if s.run.headless else 'gui'}")

    # ---------------------------------------------------------------- state
    @property
    def failures(self) -> list[Failure]:
        return [Failure.from_dict(d) for d in self.state.data.get("failures", [])]

    def _selected(self) -> list[Failure]:
        """The failures to recover, in run order (chosen once, then kept)."""
        data = self.state.data
        if "selection" not in data:
            chosen = quota.select(self.failures, self.spec.phases.quota, self.spec.phases.per_failure, self.spec.seed)
            data["selection"] = [f.id for f in chosen]
            self.state.save()
        by_id = {f.id: f for f in self.failures}
        return [by_id[i] for i in data["selection"]]

    def _job_dir(self, failure: Failure) -> Path:
        return self.out / failure.phase / failure.id

    # ---------------------------------------------------------------- stages
    def _ensure_collected(self):
        data = self.state.data
        if "failures" not in data:
            scan = self._need("scanner", self.plugins.scanner).scan(self.out)
            data["failures"], data["scan"] = [f.to_dict() for f in scan.failures], dict(scan.counts)
            self.state.save()

    def _collect(self):
        self._ensure_collected()
        data, failures = self.state.data, self.failures
        bad = sum(not f.restart.reproducible for f in failures)
        self.log(f"collect: {len(failures)} failures ({bad} unreproducible) in " + ", ".join(
            f"{n} {k}" for k, n in sorted(data["scan"].items())) + " tries")

    def _run(self):
        self._ensure_collected()
        selected, done = self._selected(), self.state.data.setdefault("jobs", {})
        runner, generator = self._need("runner", self.plugins.runner), self._need("generator", self.plugins.generator)
        todo = [(i, f) for i, f in enumerate(selected) if done.get(f.id, {}).get("status") not in ("done", "short")]
        self.log(f"run: {len(selected)} jobs selected, {len(todo)} to run")

        def attempt(item):                  # on a worker thread: touches no state
            i, f = item
            try:
                job = jb.make(i, f, self.out, self.spec.robot, self.arm)
                if not episodes.finished(episodes.read(job.dir), self.spec.phases.episodes):
                    planned = generator.plan(job.scene, self.spec.task, self.spec.robot)     # e.g. AutoMoMa's offline planner
                    if not planned.ok:
                        return {"status": "error", "reason": f"plan: {planned.reason}"}
                runner.run(job, generator)
            except JobError as e:
                return {"status": "error", "reason": str(e)}
            log = episodes.read(job.dir)
            return {"status": "done" if episodes.accepted(log) >= self.spec.phases.per_failure else "short",
                    "tries": len(log["tries"]), "accepted": episodes.accepted(log)}

        def commit(item, result):           # in job order, on this thread
            done[item[1].id] = result
            self.state.save()
            detail = result["reason"] if "reason" in result else \
                f"{result['accepted']}/{self.spec.phases.per_failure} accepted in {result['tries']} tries"
            self.log(f"run {item[1].phase} {item[1].id}: {result['status']} ({detail})")
        try:
            pool.ordered(todo, attempt, commit, self.workers, runner.stop)
        finally:
            runner.close()

    def _check(self):
        jobs = self.state.data.get("jobs", {})
        entries = self._entries()
        errors = sorted(k for k, j in jobs.items() if j["status"] == "error")
        rows = [r for r in report.table(self.spec, self.failures, self.state.data.get("selection", []), entries) if r["quota"]]
        short = [f"{r['phase']} {r['fill'] * r['quota']:.0f}/{r['quota']}" for r in rows if r["fill"] < 1.0]
        results = [("jobs", not errors, f"{len(errors)} jobs could not be run: {', '.join(errors)}" if errors else ""),
                   ("quota", not short, "episodes short of the quota: " + ", ".join(short) if short else "")]
        self.state.data["checks"] = [{"name": n, "ok": ok, "reason": why} for n, ok, why in results]
        self.state.save()
        for n, ok, why in results:
            self.log(f"check {n}: {'PASS' if ok else 'FAIL ' + why}")
        return all(ok for _, ok, _ in results)

    # ---------------------------------------------------------------- outputs
    def _entries(self) -> list[dict]:
        return mf.build(self.out, self.spec, self._job_dirs()).get("episodes", [])

    def _job_dirs(self) -> list[tuple[Failure, Path]]:
        data = self.state.data
        by_id = {f.id: f for f in self.failures}
        return [(by_id[i], self._job_dir(by_id[i])) for i in data.get("selection", []) if i in data.get("jobs", {})]

    def _report(self):
        """state.json, manifest.json (merged from the job logs, in run order) and report.md."""
        self.state.save()
        data, path = self.state.data, self.out / "manifest.json"
        manifest = mf.build(self.out, self.spec, self._job_dirs(), mf.created(path))
        mf.write(path, manifest)
        if "failures" in data:
            (self.out / "report.md").write_text(report.render(
                self.spec, self.failures, data.get("selection", []), data["scan"], manifest["episodes"], data.get("jobs", {})))

    @staticmethod
    def _need(kind, plugin):
        if plugin is None:
            raise PipelineError(f"this stage needs a {kind} plugin")
        return plugin
