"""core/generator.py: a generator that can resume gets `resume: true` unless the spec says otherwise."""
import fake_recovery as F
import pytest
from datagen.core.registry import register
from recovery.core.generator import create
from recovery.core.spec import parse


class Resumes(F.FakeGenerator):
    supports_resume = True

    def __init__(self, spec, params):
        super().__init__(spec, params)
        self.params = params


register("generator", "fake_resumes")(Resumes)


def spec(generator):
    return parse({"task": "open_fridge_door", "robot": "mobile_panda", "from": {"eval": "runs/eval/x"},
                  "phases": {"generator": generator, "quota": {"approach": 1}}, "out": "runs/recovery/x"}, "r.yaml")


def test_a_generator_that_resumes_is_created_in_resume_mode():
    assert create(spec({"plugin": "fake_resumes"})).params["resume"] is True


def test_the_spec_can_turn_resume_off():
    assert create(spec({"plugin": "fake_resumes", "resume": False})).params["resume"] is False


def test_a_generator_that_cannot_resume_gets_no_resume_param():
    g = create(spec({"plugin": "fake_recovery"}))
    assert not hasattr(g, "params") and not g.supports_resume
