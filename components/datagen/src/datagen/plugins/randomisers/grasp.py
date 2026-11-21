"""grasp: which of the planner's grasps every attempt of the seed plans (start_design.GRASPS = 5). Clustered (`method:
box`) start designs only: a reach / drive design draws its grasp point itself (reach_start / drive_start grasp_m)."""
from datagen.core.interfaces import Randomiser
from datagen.core.registry import register
from datagen.core.spec import Choices

from .design import design_for

GRASPS = 5


@register("randomiser", "grasp")
class Grasp(Randomiser):
    def limits(self, spec):
        design_for(spec, "box", who="grasp")
        return {"grasp.id": Choices(tuple(range(GRASPS)))}

    def realise(self, values, context):
        """{"grasp_id": id}; empty when `id` is not varied (the planner then cycles its grasps over the attempts)."""
        return {"grasp_id": values["id"]} if "id" in values else {}
