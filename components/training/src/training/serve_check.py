"""One inference request to a running policy server, as LeRobot 0.6.1's RobotClient (Rosetta's runner) makes it:
Ready -> SendPolicyInstructions(RemotePolicyConfig) -> SendObservations(one TimedObservation) -> GetActions.
The observation is one frame of a LeRobot dataset, in the robot's raw form (state names -> floats, camera name
-> HxWx3 uint8, task). Test support for `serve` (components/training/tests), not used by the loop.

  python serve_check.py <host:port> <checkpoint dir> <dataset root> [frame index]

Prints one JSON line {ok, n_actions, action_dim, first, latency_s}; exit 0 when ok (every value finite and the
action length equals the dataset's action feature).
"""
import json
import pickle  # nosec: our own client of our own server
import sys
import time

import grpc
import numpy as np
from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
from lerobot.datasets import LeRobotDataset
from lerobot.utils.feature_utils import hw_to_dataset_features
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks


def main(address, ckpt, root, index=0):
    ds = LeRobotDataset("revolve/dataset", root=root)
    f = ds.meta.features
    frame = ds[index]
    state_names = f["observation.state"]["names"]
    cams = [k.removeprefix("observation.images.") for k in ds.meta.camera_keys]
    hw = {**{n: float for n in state_names}, **{c: tuple(f[f"observation.images.{c}"]["shape"]) for c in cams}}
    features = hw_to_dataset_features(hw, "observation", use_video=False)
    raw = {n: float(v) for n, v in zip(state_names, frame["observation.state"].tolist())}
    for c in cams:      # dataset frames are CHW float [0, 1]; the robot sends HWC uint8
        raw[c] = (frame[f"observation.images.{c}"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    raw["task"] = frame["task"]
    stub = services_pb2_grpc.AsyncInferenceStub(grpc.insecure_channel(address, grpc_channel_options()))
    t0 = time.time()
    stub.Ready(services_pb2.Empty(), timeout=60)
    cfg = RemotePolicyConfig("smolvla", ckpt, features, actions_per_chunk=30, device="cuda")
    stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(cfg)), timeout=600)
    obs = TimedObservation(timestamp=time.time(), timestep=0, observation=raw, must_go=True)
    stub.SendObservations(send_bytes_in_chunks(pickle.dumps(obs), services_pb2.Observation, silent=True), timeout=60)
    t1 = time.time()
    data = b""
    while not data and time.time() - t1 < 120:
        data = stub.GetActions(services_pb2.Empty(), timeout=120).data
    actions = pickle.loads(data) if data else []  # nosec
    a = np.stack([np.asarray(x.get_action()) for x in actions]) if actions else np.zeros((0, 0))
    want = f["action"]["shape"][-1]
    ok = bool(len(actions)) and a.shape[-1] == want and bool(np.isfinite(a).all())
    print(json.dumps(dict(ok=ok, n_actions=len(actions), action_dim=int(a.shape[-1]) if a.size else 0,
                          dataset_action_dim=want, first=[round(float(v), 5) for v in a[0]] if a.size else [],
                          handshake_s=round(t1 - t0, 2), latency_s=round(time.time() - t1, 3))))
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        raise SystemExit(__doc__)
    sys.exit(main(*sys.argv[1:4], *map(int, sys.argv[4:])))
