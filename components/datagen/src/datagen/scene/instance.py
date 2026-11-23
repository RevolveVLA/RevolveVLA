"""A spec's scene instances: the base scene with the realisation of every randomiser that has a `scene_field`.

One instance per config: `data(config)` is the scene.yaml mapping, `realisation(config)` the replaced fields only."""
from __future__ import annotations

from datagen.core import registry
from datagen.core.interfaces import Config
from datagen.core.spec import DatagenSpec

from . import base


class Variation:
    def __init__(self, spec: DatagenSpec):
        self.spec = spec
        randomisers = {n: registry.create("randomiser", n) for n in spec.randomisers}
        self.randomisers = {n: r for n, r in randomisers.items() if r.scene_field}
        self._base = None

    def base(self) -> dict:
        if self._base is None:
            self._base = base.read_scene(self.spec)
        return self._base

    def realisation(self, config: Config) -> dict:
        """{scene key: its new value} of the config."""
        return {r.scene_field: r.realise(config.of(n), {"base": self.base()[r.scene_field]})
                for n, r in self.randomisers.items()}

    def data(self, config: Config) -> dict:
        return {**self.base(), **self.realisation(config)}

    def measured(self, config: Config) -> dict:
        """The realised start as `<randomiser>.<dim>` values (Randomiser.measure)."""
        real, base = self.realisation(config), self.base()
        return {f"{n}.{k}": v for n, r in self.randomisers.items()
                for k, v in r.measure(real[r.scene_field], base[r.scene_field]).items()}
