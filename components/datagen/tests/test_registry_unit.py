"""core/registry.py: register / get / create, lazy import by module name, errors."""
import sys
import textwrap

import pytest

from datagen.core import registry


@pytest.fixture
def clean(monkeypatch):
    monkeypatch.setattr(registry._DEFAULT, "_plugins", dict(registry._DEFAULT._plugins))


def test_register_and_get(clean):
    @registry.register("generator", "mine")
    class G:
        def __init__(self, spec, params):
            self.args = (spec, params)
    assert registry.get("generator", "mine") is G
    assert registry.create("generator", "mine", "spec", {"a": 1}).args == ("spec", {"a": 1})
    registry.register("generator", "mine")(G)          # same class again is fine
    with pytest.raises(registry.RegistryError, match="already registered"):
        registry.register("generator", "mine")(type("Other", (), {}))
    with pytest.raises(registry.RegistryError, match="unknown plugin kind"):
        registry.register("nonsense", "x")


def test_unknown_plugin_lists_what_is_known(clean):
    with pytest.raises(registry.RegistryError, match=r"no generator plugin 'nope'.*registered"):
        registry.get("generator", "nope")


def test_core_samplers_and_checks_are_found_lazily():
    assert registry.get("sampler", "random").__name__ == "RandomSampler"
    assert registry.get("check", "target").__name__ == "TargetReached"


def test_module_on_the_path_is_imported_lazily(clean, tmp_path, monkeypatch):
    (tmp_path / "lazy_plugin_mod.py").write_text(textwrap.dedent("""
        from datagen.core.registry import register
        LOADED = True
        @register("recorder", "lazy_plugin_mod")
        class R: pass
        @register("converter", "lazy_plugin_mod")
        class C: pass
    """))
    monkeypatch.syspath_prepend(str(tmp_path))
    assert "lazy_plugin_mod" not in sys.modules
    assert registry.get("recorder", "lazy_plugin_mod").__name__ == "R"
    assert registry.get("converter", "lazy_plugin_mod").__name__ == "C"       # one module, two kinds
    sys.modules.pop("lazy_plugin_mod")


def test_module_that_does_not_register_the_name(clean, tmp_path, monkeypatch):
    (tmp_path / "silent_mod.py").write_text("X = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(registry.RegistryError, match="no generator plugin 'silent_mod'"):
        registry.get("generator", "silent_mod")
    sys.modules.pop("silent_mod")


def test_core_imports_nothing_heavy():
    import subprocess
    code = ("import sys; sys.path[:0] = [sys.argv[1]]; import datagen.cli, datagen.core.pipeline;"
            "bad = [m for m in ('torch', 'automoma', 'curobo', 'rclpy', 'numpy') if m in sys.modules];"
            "assert not bad, bad")
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "src"
    subprocess.run([sys.executable, "-c", code, str(src)], check=True)
