"""Client of test_warm_reset_gpu.py, run in a client container (revolve_common.sim_client): ready, then a warm reset to each
target given as JSON (a ResetTarget), printing one RESULT JSON line with the wall time of the ready and every report."""
import json
import sys
import time

from revolve_common.sim_client import run_client
from revolve_common.warm_reset import ResetTarget


def probe(client, *targets):
    client.wait_ready()
    ready = time.time()
    reports = [client.reset_to(ResetTarget.from_json(t)).to_dict() for t in targets]
    print("RESULT " + json.dumps({"ready_ts": ready, "reports": reports}), flush=True)


if __name__ == "__main__":
    run_client(probe, sys.argv[1], *sys.argv[2:])
