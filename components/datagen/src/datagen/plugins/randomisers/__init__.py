"""Randomisers: the streams of variation a datagen spec may vary (`randomisers:` block). One module per stream,
registered under its name; core never learns what a dim means.

Adding a stream (lighting, object pose, ...) = one new module here, no core edit:

    # plugins/randomisers/lighting.py          (or any module on $DATAGEN_PLUGIN_PATH named `lighting`)
    from datagen.core.interfaces import Randomiser
    from datagen.core.registry import register
    from datagen.core.spec import Range

    @register("randomiser", "lighting")        # the name a spec uses: randomisers: {lighting: {lux: [300, 800]}}
    class Lighting(Randomiser):
        def limits(self):                      # {"<name>.<dim>": Range(lo, hi) | Choices((...))}: the widest each dim
            return {"lighting.lux": Range(100.0, 1000.0)}   # may take; spec validation rejects anything outside

        def realise(self, values, context):    # values: {dim: v} (plain dim names, `Config.of("lighting")`)
            return {"intensity": values["lux"] / 1000.0}    # -> what the generator applies; JSON-friendly, torch-free

Then a spec says `randomisers: {lighting: {lux: [300, 800]}}`; samplers, pool, manifest and checks need nothing.
The registry finds it as `datagen.plugins.randomisers.lighting` (a name that is not a module here: put it on
$DATAGEN_PLUGIN_PATH). `limits()` and `realise()` must not import torch or the generator's stack: the host samples
with them. The generator plugin calls `realise(config.of(name), context)` for each randomiser of the spec; `context`
is the generator's own data, each randomiser documents the keys it reads (robot_base: `box_centre`, `facing`).
"""
