"""C2: C1 and the handle held for at least held_fraction of the travel (the headline metric)."""
from ...core.registry import REGISTRY
from .state_log import StateLogJudge


@REGISTRY.register("judge", "c2")
class C2(StateLogJudge):
    criterion, key = "c2", "passed"
