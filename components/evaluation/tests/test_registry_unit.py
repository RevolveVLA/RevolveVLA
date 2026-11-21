"""datagen's generic Registry class and the evaluation instance."""
import sys
import textwrap
from pathlib import Path

import pytest
from datagen.core import registry as dg

from evaluation.core.registry import REGISTRY

TESTS = Path(__file__).resolve().parent


def make(**kw):
    return dg.Registry(("alpha", "policy"), "mypkg", **kw)


def test_register_get_create_per_instance():
    a, b = make(), make()

    @a.register("alpha", "one")
    class One:
        def __init__(self, x):
            self.x = x
    assert a.get("alpha", "one") is One and a.create("alpha", "one", 5).x == 5
    with pytest.raises(dg.RegistryError, match="no alpha plugin 'one'"):
        b.get("alpha", "one")                       # registries do not share plugins
    with pytest.raises(dg.RegistryError, match="unknown plugin kind"):
        a.register("beta", "x")
    with pytest.raises(dg.RegistryError, match="already registered"):
        a.register("alpha", "one")(type("Other", (), {}))
    assert issubclass(dg.RegistryError, LookupError)


def test_lookup_order_core_plugins_plural_then_bare(tmp_path, monkeypatch):
    pkg = tmp_path / "mypkg" / "plugins" / "policies"
    pkg.mkdir(parents=True)
    for d in (tmp_path / "mypkg", tmp_path / "mypkg" / "plugins", pkg):
        (d / "__init__.py").write_text("")
    (pkg / "mine.py").write_text("from datagen.core.registry import Registry\nfrom mypkg_reg import R\n"
                                 "@R.register('policy', 'mine')\nclass P: pass\n")
    (tmp_path / "mypkg" / "plugins" / "flat.py").write_text("from mypkg_reg import R\n@R.register('alpha', 'flat')\nclass F: pass\n")
    (tmp_path / "bare_mod.py").write_text("from mypkg_reg import R\n@R.register('alpha', 'bare_mod')\nclass B: pass\n")
    (tmp_path / "core_mod.py").write_text("from mypkg_reg import R\n@R.register('alpha', 'builtin')\nclass C: pass\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    reg = make(core={"alpha": "core_mod"})
    sys.modules["mypkg_reg"] = type(sys)("mypkg_reg")
    sys.modules["mypkg_reg"].R = reg
    try:
        assert reg.get("policy", "mine").__name__ == "P"            # policy -> plugins/policies/ (not `policys`)
        assert reg.get("alpha", "flat").__name__ == "F"             # <package>.plugins.<name>
        assert reg.get("alpha", "bare_mod").__name__ == "B"         # a module on sys.path
        assert reg.get("alpha", "builtin").__name__ == "C"          # the kind's core module
    finally:
        for m in ("mypkg_reg", "mypkg", "mypkg.plugins", "mypkg.plugins.policies", "mypkg.plugins.policies.mine",
                  "mypkg.plugins.flat", "bare_mod", "core_mod"):
            sys.modules.pop(m, None)


def test_extend_path_reads_the_ctor_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("MY_PLUGIN_PATH", f"{tmp_path}:{TESTS}")
    monkeypatch.setattr(sys, "path", list(sys.path))
    make(path_env="MY_PLUGIN_PATH").extend_path()
    assert sys.path[:2] == [str(tmp_path), str(TESTS)]
    before = list(sys.path)
    make().extend_path()                            # no variable: nothing added
    assert sys.path == before


def test_the_datagen_functions_are_its_default_instance():
    assert dg.register.__self__ is dg.get.__self__ is dg._DEFAULT and dg._DEFAULT.package == "datagen"
    assert dg.get("sampler", "sobol").__name__ and "generator" in dg.KINDS


def test_evaluation_registry_has_its_own_kinds_and_finds_external_plugins(monkeypatch):
    assert REGISTRY.kinds == ("ckpt", "set", "rollout", "policy", "judge") and REGISTRY.package == "evaluation"
    with pytest.raises(dg.RegistryError, match="unknown plugin kind"):
        REGISTRY.register("generator", "x")
    with pytest.raises(dg.RegistryError, match="no judge plugin 'nope'"):
        REGISTRY.get("judge", "nope")
    monkeypatch.syspath_prepend(str(TESTS))
    for kind in REGISTRY.kinds:
        assert REGISTRY.get(kind, "fake_eval").__module__ == "fake_eval"
    with pytest.raises(dg.RegistryError):
        dg.get("judge", "fake_eval")                 # not a datagen kind
