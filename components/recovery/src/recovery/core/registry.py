"""The recovery plugin registry: `REGISTRY.create("runner", "sim", spec, params)`. Plugins live in
`recovery.plugins.<kind>.<name>` or `recovery.plugins.<kind>s.<name>` (runner -> runners) or any module on $RECOVERY_PLUGIN_PATH.
(The datagen generator a recovery uses comes from datagen's own registry.)"""
from datagen.core.registry import Registry


class RecoveryRegistry(Registry):
    def _candidates(self, kind: str, name: str):
        yield f"{self.package}.plugins.{kind}.{name}"            # plugins/live/<name>.py: the kind is not pluralised
        yield from super()._candidates(kind, name)


REGISTRY = RecoveryRegistry(("runner", "scanner", "live", "expert"), "recovery", path_env="RECOVERY_PLUGIN_PATH")
