"""Rosetta's policy server, with an observation's timestep marked as inferred when the server accepts it.

LeRobot's RobotClient sends a new observation every control tick while its action queue is empty, with the same
timestep until an action runs, and only the first of them is must_go. The server marks a timestep as inferred when
the inference starts, and accepts any observation while it has not processed one yet (last_processed_obs is None).
So the observation #0 the client re-sends during the first inference is accepted as well and a second chunk
starting at timestep 0 comes back; the client aggregates the two (weighted_average), which blends chunk 0 even
with chunk_size_threshold 0. Here a timestep is marked at accept time and a non-must_go observation of a marked
timestep is dropped. Everything else is Rosetta's server.

  python -m policy_server --host H --port P --pretrained-name-or-path C --policy-device D   (as Rosetta's)
"""
import threading

import lerobot_rosetta.policy_server as base


class Server(base.RosettaPolicyServer):
    def __init__(self, config):
        super().__init__(config)
        self._enqueue_lock = threading.Lock()

    def _enqueue_observation(self, obs) -> bool:
        step = obs.get_timestep()
        with self._enqueue_lock:
            with self._predicted_timesteps_lock:
                if not obs.must_go and step in self._predicted_timesteps:
                    return False
            accepted = super()._enqueue_observation(obs)
            if accepted:
                with self._predicted_timesteps_lock:
                    self._predicted_timesteps.add(step)
        return accepted


if __name__ == "__main__":
    base.RosettaPolicyServer = Server          # base.main() builds its server from this name
    base.main()
