"""containers.run: a container docker created but could not start is removed (mocked docker client)."""
import docker
import pytest

from revolve_common import containers


class FakeContainer:
    def __init__(self, labels):
        self.labels, self.removed = labels, False

    def remove(self, force=False):
        self.removed = force


class FakeCollection:
    def __init__(self, fail):
        self.fail, self.made = fail, []

    def run(self, image, command=None, **kw):
        c = FakeContainer(kw["labels"])
        self.made.append(c)
        if self.fail:
            raise docker.errors.APIError("500 Server Error", explanation="nvidia-container-cli: nvml error: gpu requires reset")
        return c

    def list(self, all=False, filters=None):
        k, v = filters["label"].split("=")
        return [c for c in self.made if c.labels.get(k) == v and not c.removed]


class FakeDocker:
    def __init__(self, fail):
        self.containers = FakeCollection(fail)


def test_failed_start_is_removed_and_reraised():
    dk = FakeDocker(fail=True)
    other = FakeContainer({containers.LAUNCH: "someone-else"})
    dk.containers.made.append(other)
    with pytest.raises(docker.errors.APIError, match="500"):
        containers.run(dk, "img", ["true"], labels={"revolve.component": "training"}, name="t_train")
    mine = dk.containers.made[1]
    assert mine.removed and mine.labels["revolve.component"] == "training"
    assert not other.removed


def test_started_container_is_returned_untouched():
    dk = FakeDocker(fail=False)
    c = containers.run(dk, "img", ["true"])
    assert c is dk.containers.made[0] and not c.removed and containers.LAUNCH in c.labels
