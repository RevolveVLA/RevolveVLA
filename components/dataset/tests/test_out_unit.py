"""Who owns --out (dataset.cli convert and merge, DATASET-v2.md sections 3, 7, 8.5): <out> must be missing, empty or
a dataset of this component (a readable meta/revolve_dataset.json), <out>.partial missing, empty or a build directory
of this component, and neither may be or hold an input. Otherwise the command exits 1 with an `out:` problem and
touches nothing. Once the out check passes, a finished dataset at <out> that the inputs do not resume is removed
before anything can fail, so a failed rebuild never leaves the earlier dataset behind."""
import json
import os
from pathlib import Path

import pytest

from conftest import cli_main, convert_argv, fake_dataset, refused, snapshot, write_manifest
from dataset.layout import PARTIAL_MARK, partial

HAND = "/panda_hand_controller/joint_trajectory"


def refused_untouched(rc, text, out: Path, before, *also):
    """An `out:` refusal: <out> exactly as before (bytes and mtimes), no <out>.partial created; `also` = (path,
    snapshot) pairs that must be unchanged too."""
    probs = refused(rc, text, "out:")
    assert snapshot(out) == before, f"{out} was touched"
    assert not partial(out).exists(), f"{partial(out)} was created"
    for p, snap in also:
        assert snapshot(p) == snap, f"{p} was touched"
    return probs


def one_episode_run(producer, **kw):
    p = producer(**kw)
    p.episode()
    return p, p.manifest()


# ------------------------------------------------------------------ convert: what is at <out>

def test_out_is_a_file(capsys, producer, tmp_path):
    _, m = one_episode_run(producer)
    out = tmp_path / "ds"
    out.write_text("precious notes\n")
    before = snapshot(out)
    refused_untouched(*cli_main(capsys, convert_argv([m], out)), out, before)


@pytest.mark.parametrize("content", ["file", "meta_without_record"])
def test_out_is_a_foreign_nonempty_dir(capsys, producer, tmp_path, content):
    # a non-empty dir without a record is not this component's, even when it looks like a LeRobot dataset
    _, m = one_episode_run(producer)
    out = tmp_path / "ds"
    out.mkdir()
    if content == "file":
        (out / "notes.txt").write_text("mine")
    else:
        (out / "meta").mkdir()
        (out / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v3.0", "fps": 30}))
    before = snapshot(out)
    refused_untouched(*cli_main(capsys, convert_argv([m], out)), out, before)


@pytest.mark.parametrize("record", ["not json {", json.dumps({"kind": "convert"}), json.dumps(["convert"])],
                         ids=["not-json", "no-inputs", "not-an-object"])
def test_out_with_an_unreadable_record_is_foreign(capsys, producer, tmp_path, record):
    # <out> is this component's only when its record reads as one; a damaged record is someone else's directory
    _, m = one_episode_run(producer)
    out = fake_dataset(tmp_path / "ds")
    (out / "meta" / "revolve_dataset.json").write_text(record)
    before = snapshot(out)
    refused_untouched(*cli_main(capsys, convert_argv([m], out)), out, before)


def test_empty_out_dir_is_accepted(capsys, producer, tmp_path):
    # an empty dir at <out> holds nothing to lose: the run gets past the out check to the inputs (a missing result)
    p = producer()
    p.episode(result=False)
    m = p.manifest()
    out = tmp_path / "ds"
    out.mkdir()
    probs = refused(*cli_main(capsys, convert_argv([m], out)), "inputs:")
    assert not any(x.startswith("out:") for x in probs), probs
    assert not partial(out).exists()


# ------------------------------------------------------------------ convert: <out> is or holds an input

def test_out_is_the_manifest_dir(capsys, producer):
    # `--out <run>` for the run's own manifest: the manifest, bags and results stay
    p = producer()
    p.episode()
    p.episode()
    m = p.manifest()
    before = snapshot(p.root)
    refused_untouched(*cli_main(capsys, convert_argv([m], p.root)), p.root, before)


@pytest.mark.parametrize("spelling", ["prod/", "other/../prod"])
def test_out_is_the_manifest_dir_spelled_differently(capsys, producer, tmp_path, monkeypatch, spelling):
    # the comparison is on resolved paths: a relative, trailing-slash or `..` spelling of an absolute manifest's dir
    p, m = one_episode_run(producer, root=tmp_path / "prod")
    (tmp_path / "other").mkdir()
    monkeypatch.chdir(tmp_path)
    before = snapshot(p.root)
    refused(*cli_main(capsys, convert_argv([m], spelling)), "out:")
    assert snapshot(p.root) == before
    assert not (tmp_path / "prod.partial").exists()


def test_out_is_a_finished_dataset_holding_the_manifest(capsys, producer, tmp_path):
    # <out> is a dataset of this component (so not foreign) and an ancestor of the manifest's dir: the input rule wins
    # over the removal of a stale dataset
    out = fake_dataset(tmp_path / "ds")
    _, m = one_episode_run(producer, root=out / "prod")
    before = snapshot(out)
    refused_untouched(*cli_main(capsys, convert_argv([m], out)), out, before)


def test_out_is_a_finished_dataset_that_is_the_manifest_dir(capsys, producer, tmp_path):
    p, m = one_episode_run(producer, root=tmp_path / "prod")
    fake_dataset(p.root)
    before = snapshot(p.root)
    refused_untouched(*cli_main(capsys, convert_argv([m], p.root)), p.root, before)


def test_out_holds_a_bag_only(capsys, producer, tmp_path):
    # <out> holds a passed entry's bag (absolute path), while the manifest and the result live elsewhere
    out = fake_dataset(tmp_path / "ds")
    bags = producer(root=out / "bags")
    e = bags.episode(absolute=True)
    e = dict(e, result=producer(root=tmp_path / "results").episode(bag=False, absolute=True)["result"])
    m = write_manifest(tmp_path / "manifests" / "manifest.json", bags, [e])
    assert Path(e["bag"]).is_relative_to(out) and not Path(e["result"]).is_relative_to(out)
    before, mbefore = snapshot(out), snapshot(m.parent)
    refused_untouched(*cli_main(capsys, convert_argv([m], out)), out, before, (m.parent, mbefore))


def test_out_holds_a_result_only(capsys, producer, tmp_path):
    out = fake_dataset(tmp_path / "ds")
    bags = producer(root=tmp_path / "bags")
    e = bags.episode(result=False, absolute=True)
    e = dict(e, result=producer(root=out / "results").episode(bag=False, absolute=True)["result"])
    m = write_manifest(tmp_path / "manifests" / "manifest.json", bags, [e])
    assert Path(e["result"]).is_relative_to(out) and not Path(e["bag"]).is_relative_to(out)
    before = snapshot(out)
    refused_untouched(*cli_main(capsys, convert_argv([m], out)), out, before)


def test_out_is_the_bag_dir(capsys, producer):
    # <out> is a passed entry's bag dir, here also holding a record, so only the input rule refuses it
    p = producer()
    e = p.episode()
    m = p.manifest()
    bag = p.root / e["bag"]
    fake_dataset(bag)
    before = snapshot(bag)
    refused_untouched(*cli_main(capsys, convert_argv([m], bag)), bag, before)


def test_out_is_the_second_manifests_dir(capsys, producer, tmp_path):
    # every manifest counts, not only the first
    a, ma = one_episode_run(producer, root=tmp_path / "a")
    b, mb = one_episode_run(producer, root=tmp_path / "b")
    fake_dataset(b.root)
    before, abefore = snapshot(b.root), snapshot(a.root)
    refused_untouched(*cli_main(capsys, convert_argv([ma, mb], b.root)), b.root, before, (a.root, abefore))


# ------------------------------------------------------------------ convert: <out>.partial

def test_partial_is_the_manifest_dir(capsys, producer, tmp_path):
    # <out>.partial is the producer run: it is not cleared as a stale build, and <out> is not created
    out = tmp_path / "ds"
    p, m = one_episode_run(producer, root=partial(out))
    before = snapshot(p.root)
    refused(*cli_main(capsys, convert_argv([m], out)), "out:")
    assert snapshot(p.root) == before
    assert not out.exists()


def test_partial_holds_a_bag(capsys, producer, tmp_path):
    out = tmp_path / "ds"
    bags = producer(root=partial(out) / "bags")
    e = bags.episode(absolute=True)
    e = dict(e, result=producer(root=tmp_path / "results").episode(bag=False, absolute=True)["result"])
    m = write_manifest(tmp_path / "manifests" / "manifest.json", bags, [e])
    before = snapshot(partial(out))
    refused(*cli_main(capsys, convert_argv([m], out)), "out:")
    assert snapshot(partial(out)) == before
    assert not out.exists()


def test_partial_is_the_result_file(capsys, producer, tmp_path):
    # <out>.partial is a passed entry's result file (an odd but legal name)
    out = tmp_path / "ds"
    p = producer(root=tmp_path / "prod")
    e = p.episode(absolute=True)
    res = partial(out)
    os.replace(e["result"], res)
    e["result"] = str(res)
    m = p.manifest()
    before = snapshot(res)
    refused(*cli_main(capsys, convert_argv([m], out)), "out:")
    assert snapshot(res) == before
    assert not out.exists()


def test_partial_without_the_build_mark_is_foreign(capsys, producer, tmp_path):
    # a non-empty <out>.partial is replaced only when it is a build directory of this component (its mark inside)
    _, m = one_episode_run(producer)
    out = tmp_path / "ds"
    partial(out).mkdir()
    (partial(out) / "notes.txt").write_text("mine")
    before = snapshot(partial(out))
    refused(*cli_main(capsys, convert_argv([m], out)), "out:")
    assert snapshot(partial(out)) == before
    assert not out.exists()


def test_partial_with_the_build_mark_is_accepted(capsys, producer, tmp_path):
    # a kept build directory of this component passes the out check: the run goes on to the inputs (a missing result)
    p = producer()
    p.episode(result=False)
    m = p.manifest()
    out = tmp_path / "ds"
    partial(out).mkdir()
    (partial(out) / PARTIAL_MARK).touch()
    (partial(out) / "lerobot").mkdir()
    probs = refused(*cli_main(capsys, convert_argv([m], out)), "inputs:")
    assert not any(x.startswith("out:") for x in probs), probs


# ------------------------------------------------------------------ convert: a stale dataset at <out>

def test_stale_dataset_removed_when_the_manifest_fails(capsys, producer, tmp_path):
    # the manifest stage fails (a missing result) before resume is decided: nothing at <out> afterwards
    p = producer()
    p.episode(result=False)
    m = p.manifest()
    out = fake_dataset(tmp_path / "ds")
    probs = refused(*cli_main(capsys, convert_argv([m], out)), "inputs:")
    assert not any(x.startswith("out:") for x in probs), probs
    assert not out.exists(), "a stale dataset must not survive a failed rebuild"
    assert not partial(out).exists()


def test_stale_dataset_removed_when_a_bag_lacks_a_topic(capsys, producer, tmp_path):
    # the inputs are not a resume (another turn), then the bag check fails: nothing at <out> afterwards, so a caller
    # reading <out>/meta/info.json cannot take the earlier dataset
    p = producer()
    p.episode(drop=(HAND,))
    m = p.manifest()
    out = fake_dataset(tmp_path / "ds", inputs={"manifests": [str(m)], "turn": 0})
    refused(*cli_main(capsys, convert_argv([m], out, turn=1)), "inputs:")
    assert not out.exists()


# ------------------------------------------------------------------ merge

def test_merge_out_is_an_input(capsys, tmp_path):
    # `merge --out A A B`: A and B untouched
    a, b = fake_dataset(tmp_path / "A"), fake_dataset(tmp_path / "B")
    before, bbefore = snapshot(a), snapshot(b)
    refused_untouched(*cli_main(capsys, ["merge", "--out", a, a, b]), a, before, (b, bbefore))


def test_merge_out_holds_an_input(capsys, tmp_path):
    # <out> (a finished dataset itself) is an ancestor of an input: refused, not removed
    out = fake_dataset(tmp_path / "ds")
    a, b = fake_dataset(tmp_path / "A"), fake_dataset(out / "inner" / "B")
    before = snapshot(out)
    refused_untouched(*cli_main(capsys, ["merge", "--out", out, a, b]), out, before)


def test_merge_partial_is_an_input(capsys, tmp_path):
    # <out>.partial is an input (a kept build dir that also looks finished): not cleared, <out> not created
    out = tmp_path / "ds"
    a, b = fake_dataset(tmp_path / "A"), fake_dataset(partial(out))
    before = snapshot(b)
    refused(*cli_main(capsys, ["merge", "--out", out, a, b]), "out:")
    assert snapshot(b) == before
    assert not out.exists()


def test_merge_partial_holds_an_input(capsys, tmp_path):
    out = tmp_path / "ds"
    a, b = fake_dataset(tmp_path / "A"), fake_dataset(partial(out) / "lerobot" / "B")
    before = snapshot(partial(out))
    refused(*cli_main(capsys, ["merge", "--out", out, a, b]), "out:")
    assert snapshot(partial(out)) == before
    assert not out.exists()


def test_merge_out_is_a_file(capsys, tmp_path):
    a, b = fake_dataset(tmp_path / "A"), fake_dataset(tmp_path / "B")
    out = tmp_path / "merged"
    out.write_text("not yours\n")
    before = snapshot(out)
    refused_untouched(*cli_main(capsys, ["merge", "--out", out, a, b]), out, before)


def test_merge_out_is_a_foreign_dir(capsys, tmp_path):
    a, b = fake_dataset(tmp_path / "A"), fake_dataset(tmp_path / "B")
    out = tmp_path / "merged"
    out.mkdir()
    (out / "keep.txt").write_text("mine")
    before = snapshot(out)
    refused_untouched(*cli_main(capsys, ["merge", "--out", out, a, b]), out, before)


def test_merge_stale_dataset_removed_when_inputs_fail(capsys, tmp_path):
    # <out> is a finished merge of other inputs and an input is no dataset: a `merge:` problem, nothing at <out>
    a = fake_dataset(tmp_path / "A")
    out = fake_dataset(tmp_path / "merged", kind="merge", inputs={"datasets": ["elsewhere"]})
    probs = refused(*cli_main(capsys, ["merge", "--out", out, a, tmp_path / "missing"]), "merge:")
    assert not any(x.startswith("out:") for x in probs), probs
    assert not out.exists()
    assert snapshot(a) is not None
