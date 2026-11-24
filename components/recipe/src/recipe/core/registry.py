"""The recipe plugin registry: `REGISTRY.create("source", "datagen", name, params, root)`. Plugins live in
`recipe.plugins.<kinds>.<name>` (source -> sources, output -> outputs) or on $RECIPE_PLUGIN_PATH."""
from datagen.core.registry import Registry

REGISTRY = Registry(("source", "output"), "recipe", path_env="RECIPE_PLUGIN_PATH")
