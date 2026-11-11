"""One loop step inside a step container (manager image), run by revolve_common.wrapper:

  python -m revolve_common.wrapper --job-dir <step dir> -- python -m manager.steps <step dir>

job.json params (plan.step_params): `script` + `argv` = a component's own CLI (generate.py, starts.py, recover.py,
evaluate.py: their PROGRESS lines feed the wrapper's /status); dataset = dataset launch.turn_dataset; train =
training launch.start_train, its /status (step, loss) relayed as PROGRESS lines. Every container and network the
step creates through the docker SDK is labelled revolve.exp + revolve.parent=<this step container>, so the manager
can remove them when it cancels or kills the step. REVOLVE_STUB=<script.json>: tests/integration/stub_steps.py
instead of the components.
"""
from __future__ import annotations

import importlib.util
import json
import os
import runpy
import sys
import time
from pathlib import Path

from revolve_common import job as J

REPO = Path(__file__).resolve().parents[4]
RELAY = ("step", "total_steps", "loss", "eta_s", "s_per_step", "gpu_mem_gb")


def label_children(labels):
    from docker.models.containers import ContainerCollection
    from docker.models.networks import NetworkCollection
    for cls in (ContainerCollection, NetworkCollection):
        def create(self, *a, _orig=cls.create, **kw):
            kw["labels"] = {**labels, **(kw.get("labels") or {})}
            return _orig(self, *a, **kw)
        cls.create = create


def load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = sys.modules.setdefault(name, importlib.util.module_from_spec(s))
    s.loader.exec_module(m)
    return m


def train(p):
    from revolve_common import specs
    tl = load("training_launch", REPO / "components" / "training" / "launch.py")
    t = specs.Training.model_validate(p["training"])
    job = tl.start_train(Path(p["dataset"]), Path(p["out"]), mode=t.mode, base=t.base,
                         prev=Path(p["prev"]) if p["prev"] else None, steps=t.steps, batch_size=t.batch_size,
                         backend=t.backend, settings=t.backends.get(t.backend, {}), save_every=t.save_every)
    last = None
    while True:
        job.container.reload()
        s = J.read_status(job.dir)
        cur = {k: s.progress[k] for k in RELAY if s and k in s.progress}
        if cur and cur != last:
            print("PROGRESS " + json.dumps(cur), flush=True)
            last = cur
        if job.container.status not in ("created", "running"):
            break
        time.sleep(5)
    res = tl.finish(job)
    print(f"train: exit {res['exit']}, {res['wall_s']} s", flush=True)
    return 0 if res["exit"] == 0 else 1


def dataset(p):
    dl = load("dataset_launch", REPO / "components" / "dataset" / "launch.py")
    res = dl.turn_dataset(p["task"], Path(p["turn_dir"]), p["turn"], [Path(e) for e in p["episodes"]],
                          Path(p["prev_dataset"]) if p["prev_dataset"] else None)
    print(f"dataset: exit {res['exit']}, {res['wall_s']} s", flush=True)
    return 0 if res["exit"] == 0 else 1


def main(argv=None):
    job_dir = Path((argv or sys.argv[1:])[0])
    jb = J.read_job(job_dir)
    p = jb.params
    label_children({"revolve.exp": p["exp"], "revolve.parent": jb.id})   # jb.id = the step container name
    if os.environ.get("REVOLVE_STUB"):
        return load("stub_steps", REPO / "tests" / "integration" / "stub_steps.py").run(p, job_dir)
    if p["step"] == "train":
        return train(p)
    if p["step"] == "dataset":
        return dataset(p)
    script = REPO / p["script"]
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *p["argv"]]
    runpy.run_path(str(script), run_name="__main__")      # the component CLIs end in sys.exit(main())
    return 0


if __name__ == "__main__":
    sys.exit(main())
