"""What a (task, robot) pair can vary: the randomisers usable for it with their dims and allowed ranges
(`./revolve datagen --list-randomisers --task T --robot R`)."""
import pkgutil

import datagen.plugins.randomisers as package
from datagen.core import registry
from datagen.core.errors import DatagenError
from datagen.core.spec import Choices, DatagenSpec, PluginRef
from datagen.scene import base

NOT_RANDOMISERS = {"listing"}      # the other modules of this package


def usable(task: str, robot: str) -> dict:
    """{randomiser: {"<r>.<dim>": Dim}} of every randomiser of this package, for the pair."""
    spec = DatagenSpec(task, robot, "", PluginRef("-"), ())
    return {name: registry.create("randomiser", name).limits(spec)
            for name in sorted(m.name for m in pkgutil.iter_modules(package.__path__) if m.name not in NOT_RANDOMISERS)}


def render(task: str, robot: str) -> str:
    """The table: a header with the pair's scene, then a line per dim: `<randomiser>.<dim>  [lo, hi]` (or `{choices}`).
    DatagenError for a pair that does not exist."""
    try:
        scene = base.scene_name(DatagenSpec(task, robot, "", PluginRef("-"), ()))
        found = usable(task, robot)
    except (ValueError, OSError) as e:
        raise DatagenError(f"--list-randomisers: {e}") from None
    lines = [f"{task} on {robot}: scene {scene}"]
    width = max((len(k) for dims in found.values() for k in dims), default=0)
    for dims in found.values():
        for key, dim in dims.items():
            show = ("{" + ", ".join(map(str, dim.values)) + "}" if isinstance(dim, Choices)
                    else f"[{dim.lo:g}, {dim.hi:g}]")
            lines.append(f"  {key:<{width}}  {show}")
    return "\n".join(lines)
