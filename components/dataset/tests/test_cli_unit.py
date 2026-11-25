"""dataset.cli on the host (DATASET-v2.md sections 3 rule 4, 6 "Fail closed", 7, 8.4): its parser, and the input
failures, detected before anything heavy is imported, so main() returns 1 here with `inputs:` problems, the result
JSON as the last stdout line, and neither <out> nor <out>.partial created. merge refuses inputs that are not
successful builds of this component with a `merge:` problem."""
import json

import pytest

from conftest import DROP, cli_main, convert_argv, fake_dataset, last_json_line, progress_lines, refused
from dataset import cli
from dataset import manifest as M
from dataset.layout import partial


def convert(capsys, manifests, out, *extra, turn=0):
    return cli_main(capsys, convert_argv(manifests, out, *extra, turn=turn))


def fails_closed(rc, text, out, prefix="inputs:"):
    """A refusal before anything is built: the result line names <out>, nothing at <out> or <out>.partial."""
    probs = refused(rc, text, prefix)
    assert last_json_line(text)["dataset"] == str(out)
    assert not out.exists(), "nothing at <out> on a failure"
    assert not partial(out).exists(), "an input failure builds nothing (no <out>.partial)"
    return probs


def names(probs, manifest, i):
    assert any(p.startswith("inputs:") and f"{manifest} entry {i}" in p for p in probs), probs


# ------------------------------------------------------------------ convert: input failures

def test_missing_manifest(capsys, tmp_path):
    # cli: an unreadable manifest -> exit 1, inputs: problem naming it, nothing built
    m, out = tmp_path / "missing" / "manifest.json", tmp_path / "ds"
    probs = fails_closed(*convert(capsys, [m], out), out)
    assert any(str(m) in p for p in probs), probs


def test_missing_result(capsys, producer, tmp_path):
    # DATASET-v2 6 fail closed: a passed entry whose result file is missing
    p = producer()
    p.episode()
    p.episode(result=False)
    m, out = p.manifest(), tmp_path / "ds"
    names(fails_closed(*convert(capsys, [m], out), out), m, 1)


def test_missing_bag(capsys, producer, tmp_path):
    # DATASET-v2 6 fail closed: a passed entry whose bag directory (metadata.yaml) is missing
    p = producer()
    p.episode(bag="empty")
    m, out = p.manifest(), tmp_path / "ds"
    names(fails_closed(*convert(capsys, [m], out), out), m, 0)


def test_result_not_passed(capsys, producer, tmp_path):
    # DATASET-v2 6 fail closed: a result that is not passed (never re-judge: the manifest's passed is not enough)
    p = producer()
    p.episode(result_extra={"passed": False})
    m, out = p.manifest(), tmp_path / "ds"
    names(fails_closed(*convert(capsys, [m], out), out), m, 0)


def test_bag_lacks_joint_trajectory(capsys, producer, tmp_path):
    # DATASET-v2 5.4 / 8.3: a passed FollowJointTrajectory-style bag (no /<jtc>/joint_trajectory) fails the conversion
    p = producer()
    p.episode()
    p.episode(drop=("/panda_arm_controller/joint_trajectory",),
              extra_topics=("/panda_arm_controller/controller_state",))
    m, out = p.manifest(), tmp_path / "ds"
    probs = fails_closed(*convert(capsys, [m], out), out)
    names(probs, m, 1)
    assert any("/panda_arm_controller/joint_trajectory" in x for x in probs), probs


def test_bag_topic_zero_messages(capsys, producer, tmp_path):
    # decision: a contract topic with message_count 0 counts as missing
    p = producer()
    p.episode(zero=("/task/prompt",))
    m, out = p.manifest(), tmp_path / "ds"
    probs = fails_closed(*convert(capsys, [m], out), out)
    assert any("/task/prompt" in x for x in probs), probs


def test_unknown_robot(capsys, producer, tmp_path):
    # cli: an unknown robot (no specs/robots/<robot>/contract.yaml) is an inputs: problem
    p = producer(robot="no_such_robot", task="open_fridge_door", topics_of="mobile_panda")
    p.episode()
    m, out = p.manifest(), tmp_path / "ds"
    fails_closed(*convert(capsys, [m], out), out)


def test_unknown_task(capsys, producer, tmp_path):
    # cli: an unknown task (no specs/tasks/<task>/task.yaml) is an inputs: problem
    p = producer(robot="mobile_panda", task="no_such_task")
    p.episode()
    m, out = p.manifest(), tmp_path / "ds"
    fails_closed(*convert(capsys, [m], out), out)


def test_zero_passed(capsys, producer, tmp_path):
    # manifest.read: zero passed entries -> fail closed
    p = producer()
    p.episode(passed=False)
    m, out = p.manifest(), tmp_path / "ds"
    fails_closed(*convert(capsys, [m], out), out)


def test_split_without_entries(capsys, producer, tmp_path):
    # manifest.read: zero passed entries after --split -> fail closed
    p = producer()
    p.episode(split="train")
    m, out = p.manifest(), tmp_path / "ds"
    fails_closed(*convert(capsys, [m], out, "--split", "val"), out)


def test_split_filter_drops_the_bad_entry(capsys, producer, tmp_path):
    # DATASET-v2 6 --split: entries of other splits are not read at all, so a bad val entry does not matter for
    # --split train... while the train entry itself is still checked (here: its result is missing -> fail closed)
    p = producer()
    p.episode(split="train", result=False)
    p.episode(split="val", bag=False)
    m, out = p.manifest(), tmp_path / "ds"
    probs = fails_closed(*convert(capsys, [m], out, "--split", "train"), out)
    names(probs, m, 0)
    assert not any(f"{m} entry 1" in x for x in probs), probs


def test_manifests_with_different_robots(capsys, producer, tmp_path):
    # DATASET-v2 6: every manifest must have the same task and robot
    a = producer()
    a.episode()
    b = producer(robot="mobile_so101", task="open_fridge_door")
    b.episode()
    out = tmp_path / "ds"
    fails_closed(*convert(capsys, [a.manifest(), b.manifest()], out), out)


def test_entry_task_differs(capsys, producer, tmp_path):
    # DATASET-v2 6: every entry has the manifest's task
    p = producer()
    p.episode(task="open_drawer")
    m, out = p.manifest(), tmp_path / "ds"
    names(fails_closed(*convert(capsys, [m], out), out), m, 0)


def test_unlabellable_passed_entry(capsys, producer, tmp_path):
    # DATASET-v2 6 labels: neither datagen nor recovery -> fails closed (no silent skip that shifts labels)
    p = producer()
    p.episode()
    p.episode(block=None)
    m, out = p.manifest(), tmp_path / "ds"
    names(fails_closed(*convert(capsys, [m], out), out), m, 1)


def test_all_input_problems_listed(capsys, producer, tmp_path):
    # DATASET-v2 8.4: problems lists every problem (several bad entries, each named)
    p = producer()
    p.episode(result=False)
    p.episode()
    p.episode(drop=("/joint_states",))
    m, out = p.manifest(), tmp_path / "ds"
    probs = fails_closed(*convert(capsys, [m], out), out)
    names(probs, m, 0)
    names(probs, m, 2)


def test_inputs_not_modified(capsys, producer, tmp_path):
    # DATASET-v2 3 rule 5: never modify bags, results, manifests
    p = producer()
    e0 = p.episode()
    p.episode(drop=("/task/prompt",))
    m, out = p.manifest(), tmp_path / "ds"
    files = [m, p.root / e0["result"], p.root / e0["bag"] / "metadata.yaml"]
    before = [f.read_bytes() for f in files]
    fails_closed(*convert(capsys, [m], out), out)
    assert [f.read_bytes() for f in files] == before


def test_relative_paths_as_given(capsys, producer, tmp_path, monkeypatch):
    # an input problem names the manifest path (as given on the command line) and the entry index
    p = producer(root=tmp_path / "prod")
    p.episode(result=False)
    p.manifest()
    monkeypatch.chdir(tmp_path)
    rc, text = cli_main(capsys, ["convert", "--manifest", "prod/manifest.json", "--turn", "0", "--out", "ds"])
    assert rc == 1, text
    res = last_json_line(text)
    assert any(x.startswith("inputs:") and "prod/manifest.json entry 0" in x for x in res["problems"]), res
    assert not (tmp_path / "ds").exists() and not (tmp_path / "ds.partial").exists()


def test_unexpected_error_still_reports(capsys, producer, tmp_path, monkeypatch):
    # an unexpected exception in a stage: exit 1, a `failed` PROGRESS line and an `internal:` problem in the result
    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(M, "check_inputs", boom)
    if hasattr(cli, "check_inputs"):
        monkeypatch.setattr(cli, "check_inputs", boom)
    p = producer()
    p.episode()
    m, out = p.manifest(), tmp_path / "ds"
    rc, text = convert(capsys, [m], out)
    assert rc == 1, text
    assert any(x.startswith("internal: RuntimeError: boom") for x in last_json_line(text)["problems"]), text
    assert any(pr.get("stage") == "failed" for pr in progress_lines(text)), text


# ------------------------------------------------------------------ merge

def test_merge_of_non_dataset(capsys, tmp_path):
    # cli: merge of a path without meta/revolve_dataset.json -> exit 1, a merge: problem, nothing at out
    notds = tmp_path / "not_a_dataset"
    (notds / "meta").mkdir(parents=True)
    (notds / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v3.0", "fps": 30}))
    out = tmp_path / "merged"
    rc, text = cli_main(capsys, ["merge", "--out", out, notds])
    fails_closed(rc, text, out, prefix="merge:")


def test_merge_of_missing_path(capsys, tmp_path):
    # cli: merge of a path without meta/revolve_dataset.json (a path that does not exist) -> merge: problem
    out = tmp_path / "merged"
    rc, text = cli_main(capsys, ["merge", "--out", str(out), str(tmp_path / "a"), str(tmp_path / "b")])
    fails_closed(rc, text, out, prefix="merge:")


def test_merge_names_the_bad_input(capsys, tmp_path):
    # DATASET-v2 8.4: problems name what they are about
    bad = tmp_path / "plain_dir"
    bad.mkdir()
    out = tmp_path / "merged"
    rc, text = cli_main(capsys, ["merge", "--out", str(out), str(bad)])
    probs = fails_closed(rc, text, out, prefix="merge:")
    assert any(str(bad) in p for p in probs), probs


@pytest.mark.parametrize("record", [dict(problems=["frames: m.json entry 0: 640 frames, its input dataset has 600"]),
                                    dict(kind="partial"), dict(kind=DROP)],
                         ids=["failed-build", "other-kind", "no-kind"])
def test_merge_refuses_an_input_that_is_not_a_successful_build(capsys, tmp_path, record):
    # DATASET-v2 9: every input is a finished dataset of this component: a record with problems, of another kind or
    # without a kind is refused, on the records alone, before anything heavy
    good = fake_dataset(tmp_path / "good")
    other = fake_dataset(tmp_path / "also_good", kind="merge")
    bad = fake_dataset(tmp_path / "b.partial" / "lerobot" / "revolve" / "dataset", **record)
    out = tmp_path / "merged"
    probs = fails_closed(*cli_main(capsys, ["merge", "--out", out, good, other, bad]), out, prefix="merge:")
    assert any(x.startswith("merge:") and str(bad) in x for x in probs), probs
    assert not any(str(good) in x or str(other) in x for x in probs), probs


# ------------------------------------------------------------------ the command line

@pytest.mark.parametrize("argv", [["convert", "--turn", "0", "--out", "x"], ["merge", "--out", "x"], ["frobnicate"],
                                  []])
def test_bad_command_line_is_not_success(capsys, argv, tmp_path, monkeypatch):
    # DATASET-v2 7: exit 0 only when the dataset is at <out>: a broken command line never succeeds
    monkeypatch.chdir(tmp_path)
    try:
        rc = cli.main(argv)
    except SystemExit as e:              # argparse's usage error
        rc = e.code
    capsys.readouterr()
    assert rc not in (0, None)


def test_equals_forms_parse_like_two_tokens(tmp_path):
    # `--opt=V` and `--opt V` are the same command, for convert and merge
    ap = cli.parser()
    m, d = str(tmp_path / "m.json"), str(tmp_path / "ds")
    two = ap.parse_args(["convert", "--manifest", m, "--split", "train", "--turn", "0", "--out", d])
    eq = ap.parse_args(["convert", f"--manifest={m}", "--split=train", "--turn=0", f"--out={d}"])
    assert vars(two) == vars(eq)
    two, eq = ap.parse_args(["merge", "--out", d, "a", "b"]), ap.parse_args(["merge", f"--out={d}", "a", "b"])
    assert vars(two) == vars(eq)


@pytest.mark.parametrize("argv", [["convert", "--manifest", "m.json", "--turn", "0", "--ou", "ds"],
                                  ["merge", "--ou=ds", "a"]], ids=["convert", "merge"])
def test_parser_refuses_abbreviated_options(capsys, argv):
    # launch mounts exactly the paths the entrypoint opens, so an abbreviation is a usage error, never a guess
    with pytest.raises(SystemExit) as e:
        cli.parser().parse_args(argv)
    assert e.value.code == 2
    capsys.readouterr()


def test_main_refuses_an_abbreviated_option(capsys, producer, tmp_path):
    p = producer()
    p.episode(result=False)
    m, out = p.manifest(), tmp_path / "ds"
    rc, _ = cli_main(capsys, ["convert", "--manifest", m, "--turn", "0", "--ou", out])
    assert rc == 2
    assert not out.exists() and not partial(out).exists()


def test_main_accepts_equals_forms(capsys, producer, tmp_path):
    # `--manifest=M --out=D` reaches the input checks like the two-token forms: the result line names D and the
    # problem names the manifest entry
    p = producer()
    p.episode(result=False)
    m, out = p.manifest(), tmp_path / "ds"
    probs = fails_closed(*cli_main(capsys, ["convert", f"--manifest={m}", "--turn=0", f"--out={out}"]), out)
    assert any(f"{m} entry 0" in x for x in probs), probs
