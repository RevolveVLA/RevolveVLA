"""A served checkpoint's client side and service (CPU: no ROS, no torch, no docker; the policy service runs a fake model over
localhost HTTP; the real model, topics and container are the GPU test's)."""
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from datagen.core.interfaces import Config

from evaluation import policy_contract as PC
from evaluation import policy_service as PS
from evaluation import policy_wire as PW
from evaluation import rollout_client as C
from evaluation import served as S
from evaluation.core.errors import EvalError
from evaluation.core.interfaces import EvalItem
from evaluation.core.registry import REGISTRY
from evaluation.core.spec import RunnerSpec
from evaluation.plugins.policies import smolvla as P
from revolve_common.sim_fake import FakeClient, FakeSession
from revolve_common.sim_session import SimFault

c = PC.MOBILE_PANDA
JOINTS = {**{f"panda_joint{i}": 0.1 * i for i in range(1, 8)}, "panda_finger_joint1": 0.04, "panda_finger_joint2": 0.03}


# ------------------------------------------------------------------ the contract
def test_the_state_is_the_base_twist_the_arm_and_both_fingers_in_the_checkpoints_order():
    s = c.state((0.1, -0.2, 0.3), {**JOINTS, "other": 9.0})
    assert s.dtype == np.float32 and c.state_dim == 12 and c.action_dim == 11
    assert np.allclose(s, [0.1, -0.2, 0.3, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.04, 0.03])
    lone = {k: v for k, v in JOINTS.items() if k != "panda_finger_joint2"}
    assert c.state((0, 0, 0), lone)[-2:].tolist() == pytest.approx([0.04, 0.04])           # a silent mimic finger repeats


def test_an_action_goes_to_the_base_the_arm_and_the_hand_within_limits():
    pos, twist = c.action([0.1, -0.2, 0.3, 0.5, 0.5, 0.5, -1.0, 0.5, 1.0, 0.5, 0.02])
    assert twist.tolist() == [0.1, -0.2, 0.3] and pos["panda_arm_controller"].tolist() == [0.5, 0.5, 0.5, -1.0, 0.5, 1.0, 0.5]
    assert pos["panda_hand_controller"].tolist() == [0.02]
    assert c.joints == {"panda_arm_controller": [f"panda_joint{i}" for i in range(1, 8)],
                        "panda_hand_controller": ["panda_finger_joint1"]}
    pos, twist = c.action([5.0, -5.0, 5.0, *[9.0] * 7, 1.0])
    assert twist.tolist() == [0.3, -0.3, 0.6] and pos["panda_hand_controller"].tolist() == [0.04]
    assert pos["panda_arm_controller"][3] == pytest.approx(-0.0698) and pos["panda_arm_controller"][1] == pytest.approx(1.7628)


@pytest.mark.parametrize("bad", [[0.0] * 10, [np.nan] + [0.0] * 10, [np.inf] + [0.0] * 10])
def test_a_wrong_sized_or_non_finite_action_is_refused(bad):
    with pytest.raises(ValueError, match="non-finite|shape"):
        c.action(bad)


# ------------------------------------------------------------------ the observation
def fed(size=None, drop=None):
    o = S.Observer(c, size, decoder=lambda data, size: np.frombuffer(data, np.uint8).reshape(2, 2, 3) + 0)
    for i, cam in enumerate(c.cameras.values()):
        o.feed_image(cam, bytes([i] * 12))
    o.feed_joints(list(JOINTS), list(JOINTS.values()))
    o.feed_twist(0.1, 0.0, -0.1)
    return o


def test_the_observation_has_the_checkpoints_camera_names_and_the_state_vector():
    o = fed()
    assert o.ready() and o.missing() == []
    images, state = o.observe()
    assert list(images) == ["camera1", "camera2", "camera3"] and images["camera2"].shape == (2, 2, 3)
    assert [int(images[k][0, 0, 0]) for k in images] == [0, 1, 2]                          # wrist, overhead left, overhead right
    assert state.shape == (12,) and state[:3].tolist() == pytest.approx([0.1, 0.0, -0.1])


def test_the_observation_is_not_ready_until_every_topic_has_a_message():
    o = S.Observer(c, decoder=None)
    assert not o.ready() and "odom" in o.missing() and "wrist_cam" in o.missing() and "panda_joint7" in o.missing()
    with pytest.raises(RuntimeError, match="no observation.*wrist_cam"):
        o.wait_ready(0.01, sleep=lambda dt: None)
    f = fed()
    f.joints.pop("panda_joint4")
    assert not f.ready() and f.missing() == ["panda_joint4"]


# ------------------------------------------------------------------ the chunk loop
class Clock:
    """Sim time that moves only when the loop sleeps (`inference_s` per call of the policy); the result is out at `end_s`."""

    def __init__(self, end_s=2.0):
        self.t, self.end_s, self.t0 = 5.0, end_s, 5.0

    def now(self):
        return self.t

    def has_result(self, n):
        return self.end_s is not None and self.t >= self.t0 + self.end_s

    def sleep(self, dt):
        self.t += dt


class Policy:
    def __init__(self, clock, inference_s=0.2, size=50):
        self.clock, self.inference_s, self.size, self.calls = clock, inference_s, size, []

    def act(self, images, state, task):
        self.calls.append((sorted(images), state.copy(), task))
        self.clock.t += self.inference_s
        k = len(self.calls)                                      # the chunk's number is its arm's first joint, i / 100 the step in it
        return np.array([[0.01 * k, 0, 0, k + i * 0.01, *[-0.5] * 3, 0, 0, 0, 0.01] for i in range(self.size)])


def loop(clock, policy, per_chunk=20, halt=None, sent=None, **kw):
    sent = [] if sent is None else sent
    stats = S.run_chunks(clock, 0, fed(), policy, "open the fridge door", per_chunk,
                         lambda pos, twist: sent.append((pos["panda_arm_controller"][0], twist[0])), halt=halt,
                         sleep=clock.sleep, **kw)
    return stats, sent


def test_a_chunk_executes_its_first_n_actions_then_the_next_chunk_is_asked_for():
    clock = Clock(end_s=3.0)
    policy = Policy(clock)
    stats, sent = loop(clock, policy)
    assert stats["steps"] == len(sent) and stats["chunks"] == len(policy.calls) >= 4
    assert [int(p) for p, _ in sent[:21]] == [1] * 20 + [2]                                       # 20 per chunk, no more
    assert sent[0][0] == 1.0 and sent[19][0] == pytest.approx(1.19) and policy.calls[0][2] == "open the fridge door"
    assert policy.calls[0][0] == ["camera1", "camera2", "camera3"] and policy.calls[0][1].shape == (12,)
    # 0.2 s of inference per chunk, in which no action is sent: 30 Hz ticks are skipped, none sent twice
    assert 3.0 * 30 - 6 * stats["chunks"] - 10 <= stats["steps"] <= 3.0 * 30 - 6 * (stats["chunks"] - 1)
    assert stats["inference_s"] == [0.0] * stats["chunks"]                                       # wall time of the fake call


def test_the_loop_stops_when_halted_before_an_inference_and_before_each_action():
    clock, sent = Clock(end_s=3.0), []
    policy = Policy(clock)
    stats, _ = loop(clock, policy, halt=lambda: len(sent) >= 5, sent=sent)
    assert stats["steps"] == len(sent) == 5 and len(policy.calls) == 1                            # 1 chunk, 5 of its actions
    stats, sent = loop(Clock(end_s=3.0), Policy(clock), halt=lambda: True)
    assert stats["steps"] == 0 and stats["chunks"] == 0


def test_the_drive_can_be_entered_again_after_a_halt_with_a_fresh_chunk():
    clock, sent, halted = Clock(end_s=3.0), [], [False]
    policy = Policy(clock)
    first, _ = loop(clock, policy, halt=lambda: halted[0] or len(sent) >= 5, sent=sent)
    assert first["steps"] == 5 and len(policy.calls) == 1 and [int(p) for p, _ in sent] == [1] * 5
    sent.clear()
    second, _ = loop(clock, policy, sent=sent)                       # the hand-back: the same policy, the same clock, no halt
    assert len(policy.calls) == 1 + second["chunks"] and second["chunks"] >= 2
    assert [int(p) for p, _ in sent[:20]] == [2] * 20                # a new chunk from a new observation: nothing of the old one
    assert sent[0][0] == pytest.approx(2.0)                          # its first action (not the 6th of the first chunk)


def test_a_non_finite_chunk_raises_and_a_missing_result_times_out():
    clock = Clock()

    class Bad(Policy):
        def act(self, *a):
            out = super().act(*a)
            out[0, 0] = np.nan
            return out
    with pytest.raises(ValueError, match="non-finite"):
        loop(clock, Bad(clock))
    clock = Clock(end_s=None)
    with pytest.raises(SimFault, match="result_timeout"):
        loop(clock, Policy(clock), timeout_s=0.0)


def test_the_inference_rate_is_in_the_episode_file():
    s = S.summary({"steps": 40, "chunks": 2, "inference_s": [0.1, 0.3]})
    assert s["inference"] == {"n": 2, "mean_s": pytest.approx(0.2), "max_s": 0.3, "hz": pytest.approx(5.0)} and s["steps"] == 40
    assert S.summary({"steps": 0, "chunks": 0, "inference_s": []})["inference"]["hz"] is None


def test_play_served_runs_the_episode_loop_with_the_live_wrapper(tmp_path, monkeypatch):
    sess = FakeSession(tmp_path / "scene.yaml", "mobile_panda", "open_fridge_door", tmp_path / "sim", "n", 100)
    sess.start()
    monkeypatch.setattr(S.Observer, "wire", lambda self, node: self.__dict__.update(frames={}, wired=True))
    monkeypatch.setattr(S.Observer, "wait_ready", lambda self, *a, **k: None)
    obs = fed().observe()
    monkeypatch.setattr(S.Observer, "observe", lambda self: obs)
    monkeypatch.setattr(S.PolicyClient, "act", lambda self, images, state, task: np.zeros((50, 11)))
    seen, sent = [], []

    class Streamer:
        def __init__(self, node, series, lead_s):
            seen.append(("streamer", sorted(series.joints), lead_s))

        def wait_matched(self):
            seen.append("matched")

        def __call__(self, pos, twist):
            sent.append(twist)

    class Live(C.Live):
        def halt(self):
            seen.append("halt")
            return len(sent) >= 3

        def finish(self, episode):
            seen.append(("finish", episode["steps"]))
    client = FakeClient(sess)
    client.node = object()
    client.has_result = lambda n: False
    client.now = lambda t=iter(np.arange(1.0, 10.0, 0.01)): next(t)
    serve = {"url": "http://x", "prompt": "p", "robot": "mobile_panda", "actions_per_chunk": 20, "image_size": None}
    drives = []

    class Again(Live):
        def driver(self, drive):
            drives.append(drive)
            return drive
    ep = S.play_served(client, tmp_path / "r", serve, False, 1.0, None, Again(), 0.1, make_streamer=Streamer)
    sent.clear()
    again = drives[0](0)                                             # the hand-back calls the drive again mid-episode
    assert again["chunks"] == 1 and again["steps"] >= 3 and len(sent) == again["steps"]
    assert seen[:2] == [("streamer", ["panda_arm_controller", "panda_hand_controller"], 0.1), "matched"]
    assert ep["steps"] == 3 and ep["chunks"] == 1 and ("finish", 3) in seen and ep["episode"] == 0
    assert json.loads((tmp_path / "r" / "episode.json").read_text())["inference"]["n"] == 1
    with pytest.raises(ValueError, match="no observation / action contract"):
        S.play_served(client, tmp_path / "r2", {**serve, "robot": "nope"}, False, 1.0, None, Live(), 0.1)


def test_the_rollout_client_takes_a_served_policy(monkeypatch):
    got = {}
    monkeypatch.setattr(C, "run_client", lambda fn, out: got.update(fn=fn))
    monkeypatch.setattr(S, "play_served", lambda *a: got.update(args=a))
    C.main(["--sim-out", "s", "--rollout", "r", "--serve", '{"url": "u"}', "--reset"])
    got["fn"]("client")
    assert got["args"][:5] == ("client", "r", {"url": "u"}, True, C.READY_TIMEOUT_S if hasattr(C, "READY_TIMEOUT_S") else got["args"][4])
    assert got["args"][6] is C.NO_LIVE and got["args"][7] == C.LEAD_S


# ------------------------------------------------------------------ the policy service
class Model:
    info = {"chunk": 4}

    def __init__(self):
        self.seen = []

    def __call__(self, images, state, task):
        self.seen.append((sorted(images), state, task, images["camera1"].dtype))
        if task == "boom":
            raise RuntimeError("model failed")
        return np.tile(state[:3], (4, 1)) + np.arange(4)[:, None]


@pytest.fixture
def service():
    model = Model()
    srv = PS.serve(PS.Service(model), 0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield model, PW.PolicyClient(f"http://127.0.0.1:{srv.server_address[1]}", 5.0)
    srv.shutdown()
    srv.server_close()


def test_a_request_gets_the_chunk_back(service):
    model, client = service
    assert client.health()["ready"] is True and client.health()["requests"] == 0
    imgs = {f"camera{i}": np.full((4, 4, 3), i, np.uint8) for i in (1, 2, 3)}
    out = client.act(imgs, np.array([1.0, 2.0, 3.0], np.float32), "open the fridge door")
    assert out.shape == (4, 3) and out[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0] and out.dtype == np.float32
    assert model.seen[0][0] == ["camera1", "camera2", "camera3"] and model.seen[0][2] == "open the fridge door"
    assert model.seen[0][3] == np.uint8 and client.health()["requests"] == 1 and client.health()["mean_s"] >= 0


def test_a_failing_model_is_a_500_with_its_traceback_and_the_service_goes_on(service):
    _, client = service
    imgs = {"camera1": np.zeros((2, 2, 3), np.uint8)}
    with pytest.raises(RuntimeError, match="500.*model failed"):
        client.act(imgs, np.zeros(3, np.float32), "boom")
    assert client.act(imgs, np.zeros(3, np.float32), "ok").shape == (4, 3)


# ------------------------------------------------------------------ the plugin
def spec(**kw):
    return SimpleNamespace(train=SimpleNamespace(task="open_fridge_door", robot="mobile_panda"), out="runs/eval/x",
                           ckpt=SimpleNamespace(source="local", params={"path": "runs/ckpts/c"}),
                           runner=RunnerSpec(), **kw)


class FakeService:
    def __init__(self, ckpt, state_dim):
        self.ckpt, self.state_dim, self.url, self.events = ckpt, state_dim, "http://127.0.0.1:1", []

    def start(self, timeout_s):
        self.events.append(("start", timeout_s))

    def stop(self):
        self.events.append("stop")


def policy(monkeypatch, params=None, **kw):
    p = REGISTRY.create("policy", "smolvla", spec(**kw), params or {})
    monkeypatch.setattr(p, "ckpt_dir", lambda: P.REPO / "runs" / "ckpts" / "c")
    monkeypatch.setattr(p, "new_service", lambda ckpt, dim: p.__dict__.setdefault("made", FakeService(ckpt, dim)))
    monkeypatch.setattr(P.atexit, "register", lambda f: None)
    return p


def test_client_args_start_the_service_once_and_describe_the_policy(monkeypatch):
    p = policy(monkeypatch)
    item = EvalItem(Config("heldout", 501), P.REPO / "scene.yaml")
    a1, a2 = p.client_args(item), p.client_args(item)
    assert a1 == a2 and a1[0] == "--serve" and p.made.events == [("start", 300.0)] and p.made.state_dim == 12
    assert json.loads(a1[1]) == {"url": "http://127.0.0.1:1", "prompt": "open the fridge door", "robot": "mobile_panda",
                                 "actions_per_chunk": 20, "image_size": None}
    p.close()
    p.close()
    assert p.made.events == [("start", 300.0), "stop"]


def test_params_override_the_chunk_and_the_image_size_and_unknown_ones_fail(monkeypatch):
    p = policy(monkeypatch, {"actions_per_chunk": 10, "image_size": 256, "startup_s": 5})
    serve = json.loads(p.client_args(EvalItem(Config("heldout", 1), P.REPO / "s.yaml"))[1])
    assert serve["actions_per_chunk"] == 10 and serve["image_size"] == 256 and p.made.events == [("start", 5.0)]
    with pytest.raises(EvalError, match="unknown parameters.*speed"):
        REGISTRY.create("policy", "smolvla", spec(), {"speed": 1})
    other = REGISTRY.create("policy", "smolvla", SimpleNamespace(**{**vars(spec()), "train": SimpleNamespace(task="open_fridge_door", robot="x")}), {})
    with pytest.raises(EvalError, match="no observation / action contract for robot 'x'"):
        other.client_args(EvalItem(Config("heldout", 1), P.REPO / "s.yaml"))


def test_the_service_container_is_the_training_image_on_the_gpu_with_the_ckpt_read_only(tmp_path):
    ran = []

    def run(cmd, **kw):
        ran.append(cmd)
        return SimpleNamespace(returncode=0, stdout="true\n", stderr="")
    s = P.Service("ev-abc-policy", tmp_path / "runs" / "ckpts" / "c", 12, "img:1", repo=tmp_path, run=run)
    argv = s.argv()
    assert argv[:5] == ["docker", "run", "-d", "--rm", "--name"] and "--gpus" in argv and "HF_HUB_OFFLINE=1" in argv
    assert argv[argv.index("--ckpt") + 1] == "/repo/runs/ckpts/c" and argv[argv.index("--state-dim") + 1] == "12"
    assert f"{tmp_path}:/repo:ro" in argv and argv[argv.index("img:1") + 1:argv.index("img:1") + 4] == ["python", "-m", "evaluation.policy_service"]
    health = iter([OSError("refused"), {"ready": False}, {"ready": True}])

    class Client:
        def health(self):
            h = next(health)
            if isinstance(h, Exception):
                raise h
            return h
    s.start(10, client=Client(), sleep=lambda dt: None)
    assert ran[0] == argv and (tmp_path / "cache" / "hf").is_dir()
    s.stop()
    assert ran[-1][:3] == ["docker", "rm", "-f"]


def test_a_service_that_dies_or_never_loads_is_an_error_with_its_log(tmp_path):
    state = {"running": "false"}

    def run(cmd, **kw):
        out = {"inspect": state["running"] + "\n", "logs": "CUDA out of memory"}.get(cmd[1], "")
        return SimpleNamespace(returncode=0, stdout=out, stderr="")
    s = P.Service("n", tmp_path / "runs" / "c", 12, repo=tmp_path, run=run)

    class Down:
        def health(self):
            raise OSError("refused")
    with pytest.raises(EvalError, match="exited:\nCUDA out of memory"):
        s.start(10, client=Down(), sleep=lambda dt: None)
    state["running"] = "true"
    with pytest.raises(EvalError, match="not ready after 0 s:\nCUDA"):
        s.start(0, client=Down(), sleep=lambda dt: None)
