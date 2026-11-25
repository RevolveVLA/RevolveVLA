"""Stand-in for the manager: hands the dataset component a convert or merge job and waits for its result.

    python components/dataset/mock_manager.py convert --manifest M [--manifest M2 ...] [--split S] --turn K --out D
    python components/dataset/mock_manager.py merge --out D DATASET [DATASET ...]
    [--job-dir J] before the job kind: the job directory (default <D's parent>/jobs/dataset_<kind>_<D's name>)

The job runs as the manager will run it: the entrypoint of `./revolve dataset` (src/dataset/cli.py) under the HTTP job
wrapper in the dataset image (launch.start: job.json in; status.json, result.json, stdout.log out). Prints result.json;
exits with the job's code. What to convert or merge, and the turn, are the caller's choice: none of it is decided
here."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1] / "common")]
_spec = importlib.util.spec_from_file_location("dataset_launch", HERE / "launch.py")
sys.modules["dataset_launch"] = launch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launch)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mock_manager.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--job-dir", help="the job directory")
    ap.add_argument("job", nargs=argparse.REMAINDER, help="convert ... | merge ... (as for ./revolve dataset)")
    a = ap.parse_args(argv)
    try:
        job = launch.start(a.job, a.job_dir)
    except launch.LaunchError as e:
        ap.error(str(e))
    res = launch.wait(job)
    print(json.dumps(res["result"], indent=1))
    return res["exit"]


if __name__ == "__main__":
    sys.exit(main())
