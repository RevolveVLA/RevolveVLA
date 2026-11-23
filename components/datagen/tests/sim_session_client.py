"""Client of test_sim_session_gpu.py, run in a client container (revolve_common.sim_client): ready, then N episodes
with a reset between them and no commands (each ends on a timeout), printing one RESULT JSON line."""
import json
import sys

from revolve_common.sim_client import run_client


def episodes(client, n):
    client.wait_ready()
    out = []
    for i in range(n):
        if i:
            client.reset()
        k = client.start_episode()
        out.append({"episode": k, "result": client.wait_result(k), "files": client.wait_files(k)})
    print("RESULT " + json.dumps({"episodes": out}), flush=True)


if __name__ == "__main__":
    run_client(episodes, sys.argv[1], int(sys.argv[2]))
