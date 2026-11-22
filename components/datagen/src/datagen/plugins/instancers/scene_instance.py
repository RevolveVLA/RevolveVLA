"""instancer scene_instance: writes <dir>/scene.yaml, the base scene with each scene randomiser's realisation
(`robot_start`), outside specs/ (DATAGEN-v2.md 8.1); `name` and `assets` stay as in the base scene.

The file loads through revolve_common.specs.load and passes `./revolve validate <file>`."""
from __future__ import annotations

from pathlib import Path

import yaml

from datagen.core.interfaces import Config, Instancer, Result
from datagen.core.registry import register
from datagen.scene.instance import Variation


@register("instancer", "scene_instance")
class SceneInstance(Instancer):
    def __init__(self, spec, params):
        self.variation = Variation(spec)

    def write(self, config: Config, out: Path) -> Result:
        out.mkdir(parents=True, exist_ok=True)
        path = out / "scene.yaml"
        text = yaml.safe_dump(self.variation.data(config), sort_keys=False, default_flow_style=None)
        path.write_text(f"# scene instance: seed {config.seed} of split {config.split} (datagen)\n{text}")
        return Result(realised=self.realised(config), paths={"scene": str(path)})

    def realised(self, config: Config):
        return self.variation.measured(config)
