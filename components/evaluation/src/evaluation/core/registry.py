"""The evaluation plugin registry: `@REGISTRY.register("ckpt", "local")`, `REGISTRY.create("judge", "c2", spec, params)`.
Plugins live in `evaluation.plugins.<kind>s.<name>` (`policy` -> `policies`) or any module on $EVAL_PLUGIN_PATH."""
from datagen.core.registry import Registry

REGISTRY = Registry(("ckpt", "set", "rollout", "policy", "judge", "training"), "evaluation", path_env="EVAL_PLUGIN_PATH")
