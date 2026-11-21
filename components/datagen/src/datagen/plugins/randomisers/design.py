"""The start design a randomiser reads its limits from: task.yaml starts.<robot> (specs_bridge.starts) of the spec's
task and robot, at the generator's `variant` when it names one. One randomiser per start method."""
from datagen.core.errors import DatagenError

BY_METHOD = {"box": "robot_base", "reach": "reach_start", "drive": "drive_start"}    # the randomiser of each method


class StartDesignError(DatagenError):
    pass


def variant_of(spec) -> str | None:
    """The datagen variant of the spec's generator (its `variant` param), None for the task's own block."""
    return spec.generator.params.get("variant")


def design_for(spec, method: str, who: str | None = None):
    """The `method` start design of spec.robot on spec.task (specs.BoxStarts / ReachStarts / DriveStarts); a
    StartDesignError naming the right randomiser when the robot has another method or none. who: the randomiser asking
    (default: the method's own)."""
    who = who or BY_METHOD[method]
    from datagen.specs_bridge import starts            # reads the task spec (revolve_common); not for realise
    design = starts(spec.task, spec.robot, variant_of(spec))
    if design is None:
        raise StartDesignError(f"{who}: {spec.robot} on {spec.task} has no start design "
                               f"(task.yaml starts.{spec.robot}); it needs `method: {method}`")
    if design.method != method:
        raise StartDesignError(f"{spec.robot} on {spec.task} uses {design.method}: use {BY_METHOD[design.method]}, "
                               f"not {who}")
    return design


def any_design(spec, who: str):
    """The start design of spec.robot on spec.task whatever its method; a StartDesignError without one."""
    from datagen.specs_bridge import starts
    design = starts(spec.task, spec.robot, variant_of(spec))
    if design is None:
        raise StartDesignError(f"{who}: {spec.robot} on {spec.task} has no start design (task.yaml starts.{spec.robot})")
    return design
