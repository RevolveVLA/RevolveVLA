"""What a (task, robot) pair can vary: the randomisers usable for it with their dims and allowed ranges
(`./revolve datagen --list-randomisers --task T --robot R [--variant V]`)."""
import pkgutil

import datagen.plugins.randomisers as package
from datagen.core import registry
from datagen.core.errors import DatagenError
from datagen.core.spec import Choices, DatagenSpec, PluginRef

from .design import StartDesignError

NOT_RANDOMISERS = {"design", "listing"}      # the other modules of this package


def usable(task: str, robot: str, variant: str | None = None) -> dict:
    """{randomiser: {"<r>.<dim>": Dim}} of every randomiser of this package whose start design fits the pair."""
    spec = DatagenSpec(task, robot, "", PluginRef("automoma", {"variant": variant} if variant else {}), ())
    out = {}
    for name in sorted(m.name for m in pkgutil.iter_modules(package.__path__) if m.name not in NOT_RANDOMISERS):
        try:
            out[name] = registry.create("randomiser", name).limits(spec)
        except StartDesignError:
            continue
    return out


def render(task: str, robot: str, variant: str | None = None) -> str:
    """The table: a header with the pair's start method, then one line per dim: `<randomiser>.<dim>  [lo, hi]` (or
    `{choices}`). DatagenError for a pair or variant that does not exist."""
    from datagen.specs_bridge import starts            # reads the task spec (revolve_common)
    try:
        design = starts(task, robot, variant)
        found = usable(task, robot, variant)
    except (ValueError, OSError) as e:
        raise DatagenError(f"--list-randomisers: {e}") from None
    lines = [f"{task} on {robot}" + (f" (variant {variant})" if variant else "")
             + f": start design {design.method if design else 'none'}"]
    width = max((len(k) for dims in found.values() for k in dims), default=0)
    for dims in found.values():
        for key, dim in dims.items():
            show = "{" + ", ".join(map(str, dim.values)) + "}" if isinstance(dim, Choices) else f"[{dim.lo:g}, {dim.hi:g}]"
            lines.append(f"  {key:<{width}}  {show}")
    return "\n".join(lines)
