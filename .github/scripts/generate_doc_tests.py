"""Generate ignored pytest modules from Python examples in Markdown files.

The CLI uses ``fire``, which ships in the ``cli`` extra::

    pip install "vision-synth[cli]"

Usage::

    python .github/scripts/generate_doc_tests.py

"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from phmdoctest.direct import Marker
from phmdoctest.fenced import Role, convert_nodes
from phmdoctest.fillrole import identify_code_output_session_blocks
from phmdoctest.tool import detect_python_examples, fenced_block_nodes

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "tests" / "integration"


def _markdown_sources() -> list[Path]:
    """Return README and published documentation files containing Python examples.

    ``docs/adr/`` is excluded from the site (``exclude_docs`` in ``mkdocs.yml``), so its examples are not tested either.

    """
    docs = ROOT / "docs"
    candidates = [ROOT / "README.md", *sorted(p for p in docs.rglob("*.md") if docs / "adr" not in p.parents)]
    return [path for path in candidates if (examples := detect_python_examples(path)).has_code or examples.has_session]


def _output_path(markdown: Path) -> Path:
    """Map a Markdown source path to a stable generated pytest path."""
    relative = markdown.relative_to(ROOT)
    if relative == Path("README.md"):
        return OUTPUT_DIR / "test_readme.py"
    flattened = "__".join(relative.with_suffix("").parts)
    return OUTPUT_DIR / f"test_{flattened}.py"


def _print_blocks_without_output(markdown: Path) -> list[int]:
    """Return lines of Python blocks that print without expected output."""
    with markdown.open(encoding="utf-8") as source:
        blocks = convert_nodes(fenced_block_nodes(source))
    identify_code_output_session_blocks(blocks)
    return [
        block.line
        for block in blocks
        if block.role == Role.CODE
        and "print(" in block.contents
        and (not block.output or block.output.role != Role.OUTPUT)
        and not block.has_directive(Marker.SKIP)
    ]


def main() -> None:
    """Regenerate pytest files for all Python examples in project Markdown."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for generated in OUTPUT_DIR.glob("test_*.py"):
        generated.unlink()

    sources = _markdown_sources()
    missing_output = {markdown: lines for markdown in sources if (lines := _print_blocks_without_output(markdown))}
    if missing_output:
        details = "; ".join(
            f"{path.relative_to(ROOT)}:{', '.join(map(str, lines))}" for path, lines in missing_output.items()
        )
        raise RuntimeError(f"Python print examples need expected-output blocks: {details}")

    for markdown in sources:
        output = _output_path(markdown)
        subprocess.run(  # noqa: S603 - the command and paths are project-controlled
            [
                sys.executable,
                "-m",
                "phmdoctest",
                str(markdown.relative_to(ROOT)),
                "-s",
                "phmdoctest:skip",
                "--outfile",
                str(output.relative_to(ROOT)),
            ],
            check=True,
            cwd=ROOT,
        )

    print(f"Generated {len(sources)} documentation test modules in {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
