"""Static, import-free proof that only `datasets.py` pulls torch or fused_transforms into `synth_datasets`.

Both existing guards for this boundary are runtime and deletable: `test_no_torch_import.py` imports the package in a
subprocess and checks `sys.modules`, and the CI `synth-datasets-no-torch` leg silences `datasets.py`'s own torch import
with a `--ignore=src/synth_datasets/export/datasets.py` flag rather than a rule. Neither would notice a *new* module
quietly gaining a torch or fused_transforms import until someone happens to run a torch-full test session, or the
`--ignore` flag is remembered to be updated. This walks the AST of every module instead, so the boundary holds by
construction and works with torch absent.

"""

from __future__ import annotations

import ast
from pathlib import Path

import synth_datasets

_FORBIDDEN_PREFIXES = ("torch", "fused_transforms")
_EXEMPT_MODULE = "datasets.py"


def _imported_forbidden_names(source_path: Path) -> list[str]:
    """Return every dotted import name in `source_path` that starts with a forbidden prefix."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names if alias.name.startswith(_FORBIDDEN_PREFIXES))
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(_FORBIDDEN_PREFIXES):
            names.append(node.module)
    return names


def _collect_violations() -> dict[str, list[str]]:
    """Map each non-exempt `synth_datasets` module (relative path) to its forbidden imports, for those that have
    any."""
    package_root = Path(synth_datasets.__file__).parent
    violations: dict[str, list[str]] = {}
    for path in sorted(package_root.rglob("*.py")):
        if path.name == _EXEMPT_MODULE:
            continue
        found = _imported_forbidden_names(path)
        if found:
            violations[str(path.relative_to(package_root))] = found
    return violations


def test_no_module_outside_datasets_imports_torch_or_fused_transforms() -> None:
    """Every `synth_datasets` module except `datasets.py` is free of torch and fused_transforms import nodes.

    Parses source with `ast` and never executes it, so this holds whether or not torch is installed, and it would catch
    a new torch/fused_transforms import the moment it lands in a module other than `datasets.py` — the one place that
    import is allowed today.

    """
    violations = _collect_violations()
    assert violations == {}, f"forbidden import(s) found outside {_EXEMPT_MODULE}: {violations}"
