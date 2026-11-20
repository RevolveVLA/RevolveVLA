"""grasp: which of the planner's grasps every attempt of the seed plans (start_design.GRASPS = 5)."""
from datagen.core.interfaces import Randomiser
from datagen.core.registry import register
from datagen.core.spec import Choices

GRASPS = 5


@register("randomiser", "grasp")
class Grasp(Randomiser):
    def limits(self):
        return {"grasp.id": Choices(tuple(range(GRASPS)))}

    def realise(self, values, context):
        """{"grasp_id": id}; empty when `id` is not varied (the planner then cycles its grasps over the attempts)."""
        return {"grasp_id": values["id"]} if "id" in values else {}
