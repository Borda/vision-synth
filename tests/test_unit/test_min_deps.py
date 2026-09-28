"""Regression tests for `.github/scripts/min_deps.py`'s `_replace_min_versions` pinning logic.

`min_deps.py` is not a package (`testpaths` excludes `.github`), and its module docstring's doctest never runs for the
same reason, so `_replace_min_versions` had zero executable coverage despite this PR modifying it to pin new extras. A
regression here would silently miswrite the oldest-dependencies CI leg. These tests load the script by path — the
established pattern this project already uses for `experiments/` scripts in `tests/test_unit/test_benchmarks.py` — and
drive it against a throwaway fixture `pyproject.toml`.

Known gap: `tomlkit` (what `min_deps.py` itself imports) is commented out of the `dev` dependency group in
`pyproject.toml` — only the CI `oldest`-deps leg installs it, ad hoc, before invoking `min_deps.py` directly as a
script. This module is scoped to `tests/` only, so it cannot add `tomlkit` to `dev` itself; it guards with
`pytest.importorskip` instead. That means these tests run for real wherever `tomlkit` happens to be installed
(including, today, the `oldest`-deps leg's own environment right after that install step) and skip cleanly everywhere
else, rather than failing collection for every other CI leg and local dev checkout.

"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

tomlkit = pytest.importorskip(
    "tomlkit",
    reason="min_deps.py needs tomlkit; the oldest-deps CI leg installs it on demand",
    exc_type=ModuleNotFoundError,
)

_ROOT = Path(__file__).resolve().parents[2]

_FIXTURE_TOML = """\
[project]
name = "demo"
dependencies = [
  "numpy>=1.26",
  "pyyaml==6.0",
]

[project.optional-dependencies]
torch = [
  "torch>=2.2",
]
extra = [
  "requests<3",
]

[dependency-groups]
dev = [
  "pytest>=8.3,<8.5",
  {include-group = "docs"},
]
"""


def _load_min_deps() -> ModuleType:
    """Load `.github/scripts/min_deps.py` by path, without adding it to the package API (it is not a package)."""
    path = _ROOT / ".github" / "scripts" / "min_deps.py"
    spec = importlib.util.spec_from_file_location("min_deps", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["min_deps"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pyproject_file(tmp_path: Path) -> Path:
    """Write `_FIXTURE_TOML` under `tmp_path` and return its path."""
    path = tmp_path / "pyproject.toml"
    path.write_text(_FIXTURE_TOML)
    return path


def test_pins_every_ge_specifier_across_all_three_sections(pyproject_file: Path) -> None:
    """Every `>=` requirement in dependencies, optional-dependencies, and dependency-groups is pinned to `==`.

    The fixture covers all three sections `_replace_min_versions` walks (`[project.dependencies]`, `[project.optional-
    dependencies]`, `[dependency-groups]`), each with one `>=` requirement, so a regression that stops walking any one
    section shows up as a shorter `changed` list.

    """
    min_deps = _load_min_deps()

    changed = min_deps._replace_min_versions(str(pyproject_file))

    assert sorted(changed) == sorted(["numpy==1.26", "torch==2.2", "pytest==8.3,<8.5"])


def test_leaves_non_ge_specifiers_untouched(pyproject_file: Path) -> None:
    """A requirement without `>=` (already exact-pinned, or a different operator) is not rewritten.

    `pyyaml==6.0` and `requests<3` sit alongside `>=` requirements in the same fixture sections; this confirms the `">="
    in req` guard is selective rather than rewriting every requirement it walks past.

    """
    min_deps = _load_min_deps()

    min_deps._replace_min_versions(str(pyproject_file))

    doc = tomlkit.parse(pyproject_file.read_text())
    assert doc["project"]["dependencies"][1] == "pyyaml==6.0"
    assert doc["project"]["optional-dependencies"]["extra"][0] == "requests<3"


def test_skips_dependency_group_include_group_references(pyproject_file: Path) -> None:
    """A PEP 735 `{include-group = ...}` table entry in a dependency group is left untouched, not treated as a string.

    `_replace_min_versions` guards its dependency-groups loop with `isinstance(req, str)` specifically for
    this shape (a table reference to another group rather than a requirement string); this pins that the
    guard actually prevents the table entry from being rewritten or dropped.

    """
    min_deps = _load_min_deps()

    min_deps._replace_min_versions(str(pyproject_file))

    doc = tomlkit.parse(pyproject_file.read_text())
    assert doc["dependency-groups"]["dev"][1] == {"include-group": "docs"}
