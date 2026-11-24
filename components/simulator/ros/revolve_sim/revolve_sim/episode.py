"""The phase logic of one episode (no ROS): ordered, state-driven phases of a task (revolve_common.specs).

Each step(t, values) gets the sim time and the value of every condition (keys from keys()):
  1. sanity checks: an untimed one (`timed: false`) that is violated fails the episode as phase `global`; a timed
     one fails the active phase;
  2. the active phase's `while` conditions: a violated one fails the phase;
  3. all of the active phase's `done` conditions true (each for its for_s) -> the phase passes, the next one starts
     (the last one: the episode passes);
  4. timeout_s of the active phase, then the episode's budget_s: passed -> the active phase fails.
A value that is None or NaN (not computable yet) is never true and never a violation. A `for_s` on a `while` or
sanity condition is the time the violation has to last.
"""
import math
import re
from pathlib import Path

from revolve_common import specs

from revolve_sim.metrics import compare

EPS = 1e-9
# the files of episode n in a run dir: its result, its gz log copy, its bag (also a copy rosbag2 renamed: _bag(1))
EPISODE_FILE = re.compile(r'episode_(\d+)(\.json|_gz\.log|_bag.*)')


def next_episode(run_dir):
    """The number of the next episode in run_dir: one past the highest n of any episode_<n> file (0 if none). A bag
    without a result (a run killed mid-episode) counts too, so a relaunch into the same dir overwrites nothing."""
    p = Path(run_dir)
    found = [int(m.group(1)) for f in (p.iterdir() if p.is_dir() else []) if (m := EPISODE_FILE.fullmatch(f.name))]
    return max(found, default=-1) + 1


def describe(c):
    params = ' '.join(f'{k}={v}' for k, v in c.params.items() if k != 'ignore')
    return f'{c.metric} {params} {c.op} {c.value:g}'.replace('  ', ' ') + (f' for {c.for_s:g} s' if c.for_s else '')


class Hold:
    """How long a condition has been true (done) or violated (while / sanity)."""

    def __init__(self, cond, violation):
        self.cond, self.violation, self.since = cond, violation, None

    def update(self, t, value):
        known = value is not None and math.isfinite(value)
        hit = compare(self.cond.op, value, self.cond.value)
        if self.violation:
            hit = known and not hit
        if not hit:
            if not (self.violation and not known):     # an unknown value keeps a violation's timer running
                self.since = None
            return False
        if self.since is None:
            self.since = t
        return t - self.since >= (self.cond.for_s or 0.0) - EPS


def _round(v):
    return None if v is None or not math.isfinite(v) else round(float(v), 6)


class Episode:
    def __init__(self, task, sanity, t0):
        self.task, self.sanity, self.t0 = task, dict(sanity), t0
        self.i = 0
        self.result = None
        self.timeline = [{'name': p.name, 'status': 'not_reached', 't_start': None, 't_end': None, 'last': {}}
                         for p in task.phases]
        self.sanity_holds = {f'sanity.{k}': Hold(c, True) for k, c in self.sanity.items()}
        self.last_sanity = {}
        self._start_phase(t0)

    # keys: '<phase>.done[i]', '<phase>.while[i]', 'sanity.<check>'
    @staticmethod
    def phase_keys(phase):
        return ([(f'{phase.name}.done[{i}]', c) for i, c in enumerate(phase.done)]
                + [(f'{phase.name}.while[{i}]', c) for i, c in enumerate(phase.while_)])

    def keys(self):
        """[(key, condition)] to evaluate now: the active phase's and the sanity checks."""
        if self.result is not None:
            return []
        return [*self.phase_keys(self.task.phases[self.i]), *((k, h.cond) for k, h in self.sanity_holds.items())]

    @property
    def phase(self):
        return None if self.result is not None else self.task.phases[self.i].name

    def _start_phase(self, t):
        p = self.task.phases[self.i]
        self.phase_start = t
        self.timeline[self.i].update(status='active', t_start=round(t - self.t0, 4))
        self.holds = {k: Hold(c, k.split('.')[-1].startswith('while')) for k, c in self.phase_keys(p)}

    def _end(self, t, passed, phase=None, reason=None, check=None, cond=None, value=None):
        rel = round(t - self.t0, 4)
        if phase is not None:          # `global`: the active phase is aborted
            self.timeline[self.i].update(status='failed' if phase != specs.GLOBAL_PHASE else 'aborted', t_end=rel,
                                         reason=reason)
        self.result = {'passed': passed, 'failed_phase': phase, 'reason': reason, 't_fail': None if passed else rel,
                       'check': check, 'condition': describe(cond) if cond is not None else None,
                       'value': _round(value), 'duration_s': rel, 'phases': self.timeline,
                       'sanity_last': self.last_sanity}
        return self.result

    def step(self, t, values):
        """Advance to sim time t with {key: value}; -> the result dict once the episode has ended, else None."""
        if self.result is not None:
            return self.result
        p = self.task.phases[self.i]
        entry = self.timeline[self.i]
        entry['last'] = {k: _round(values.get(k)) for k in self.holds}
        self.last_sanity = {k: _round(values.get(k)) for k in self.sanity_holds}
        bad = [(k, h) for k, h in self.sanity_holds.items() if h.update(t, values.get(k))]
        for timed in (False, True):
            for k, h in bad:
                if h.cond.timed == timed:
                    return self._end(t, False, p.name if timed else specs.GLOBAL_PHASE, 'sanity', k, h.cond,
                                     values.get(k))
        done = True
        for k, h in self.holds.items():
            hit = h.update(t, values.get(k))
            if h.violation and hit:
                return self._end(t, False, p.name, 'while', k, h.cond, values.get(k))
            if not h.violation:
                done = done and hit
        if done:
            entry.update(status='passed', t_end=round(t - self.t0, 4))
            if self.i + 1 == len(self.task.phases):
                return self._end(t, True)
            self.i += 1
            self._start_phase(t)
            return None
        if p.timeout_s is not None and t - self.phase_start > p.timeout_s + EPS:
            return self._end(t, False, p.name, 'timeout_s', value=t - self.phase_start)
        if t - self.t0 > self.task.budget_s + EPS:
            return self._end(t, False, p.name, 'budget_s', value=t - self.t0)
        return None
