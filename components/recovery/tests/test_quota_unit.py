"""core/quota.py: per-phase counts, round robin over configs, interleaved phases, determinism, unreproducible left out."""
from recovery.core import quota as Q
from recovery.core.failures import Failure
from recovery.core.restart import Restart


def failure(config, n, phase, ok=True, set="heldout"):
    return Failure(f"{set}_{config}_r{n}_t0", set, f"{config}_r{n}", 0, f"/runs/{set}/scene_{config}/scene.yaml", phase, "verdict", "",
                   10.0, 0.0, "bag", "result", Restart(9.0, (0.0, 0.0, 0.0), (0.0,), 0.0, None if ok else "door"))


FAILS = [failure(c, n, p) for c in (1, 2, 3) for n in (0, 1) for p in ("approach", "grasp")]


def ids(fs):
    return [f.id for f in fs]


def test_counts_follow_the_quota_and_per_failure():
    chosen = Q.select(FAILS, {"approach": 4, "grasp": 2}, 1, 0)
    assert sorted((f.phase for f in chosen)) == ["approach"] * 4 + ["grasp"] * 2
    assert len(Q.select(FAILS, {"approach": 5}, 2, 0)) == 3          # 5 episodes at 2 per failure: 3 failures
    assert Q.select(FAILS, {"approach": 0, "grasp": 0}, 1, 0) == []
    assert Q.select(FAILS, {}, 1, 0) == []


def test_short_phases_give_what_there_is():
    assert len(Q.select(FAILS, {"approach": 100}, 1, 0)) == 6
    assert Q.select(FAILS, {"open": 3}, 1, 0) == []


def test_round_robin_over_configs_then_interleaved_phases():
    chosen = Q.select(FAILS, {"approach": 3, "grasp": 3}, 1, 0)
    approach, grasp = [f for f in chosen if f.phase == "approach"], [f for f in chosen if f.phase == "grasp"]
    assert sorted(f.rollout.split("_")[0] for f in approach) == ["1", "2", "3"]       # one per config before a config has two
    assert sorted(f.rollout.split("_")[0] for f in grasp) == ["1", "2", "3"]
    assert [f.phase for f in chosen] == ["approach", "grasp"] * 3


def test_second_round_takes_the_next_failure_of_each_config():
    chosen = Q.select(FAILS, {"approach": 4}, 1, 0)
    assert len({f.rollout.split("_")[0] for f in chosen[:3]}) == 3
    assert {f.rollout.split("_")[1] for f in chosen[:3]} == {"r0"}


def test_deterministic_and_seeded():
    a = ids(Q.select(FAILS, {"approach": 2, "grasp": 2}, 1, 0))
    assert a == ids(Q.select(FAILS, {"approach": 2, "grasp": 2}, 1, 0))
    orders = {tuple(ids(Q.select(FAILS, {"approach": 3}, 1, seed))) for seed in range(8)}
    assert len(orders) > 1


def test_unreproducible_failures_are_not_selected():
    fails = [failure(1, 0, "approach", ok=False), failure(2, 0, "approach"), failure(3, 0, "approach", ok=False)]
    assert ids(Q.select(fails, {"approach": 3}, 1, 0)) == [fails[1].id]


def test_a_config_is_its_set_and_scene():
    assert Q.config_of(failure(1, 0, "approach")) != Q.config_of(failure(1, 0, "approach", set="train"))
    assert Q.config_of(failure(1, 0, "approach")) == Q.config_of(failure(1, 1, "grasp"))


def test_admission_picks_failures_as_they_arrive_up_to_the_phases_jobs():
    adm = Q.Admission({"approach": 3, "grasp": 1}, 2)                    # 2 approach jobs, 1 grasp job
    first = adm.pick([failure(1, 0, "approach"), failure(1, 1, "grasp"), failure(1, 2, "approach", ok=False), failure(2, 0, "open")])
    assert ids(first) == ["heldout_1_r0_t0", "heldout_1_r1_t0"]          # an unreproducible one or a phase out of the quota is left
    second = adm.pick([failure(2, 1, "approach"), failure(2, 2, "approach"), failure(2, 3, "grasp")])
    assert ids(second) == ["heldout_2_r1_t0"]                            # the phases are full
    assert [f.phase for f in Q.Admission({"approach": 3}, 2, ["approach", "grasp"]).pick(FAILS)] == ["approach"]       # jobs of an earlier run count


def test_admission_in_one_batch_equals_the_head_of_each_phase():
    one = Q.Admission({"approach": 2}, 1).pick(FAILS)
    assert ids(one) == ids([f for f in FAILS if f.phase == "approach"][:2])
