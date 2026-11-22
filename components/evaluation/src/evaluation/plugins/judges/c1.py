"""C1: the door at or above the task's joint_deg for hold_s, by any means (a push counts)."""
from ...core.registry import REGISTRY
from .state_log import StateLogJudge


@REGISTRY.register("judge", "c1")
class C1(StateLogJudge):
    criterion, key = "c1", "opened"
