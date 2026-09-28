"""The ``vision-synth`` console script, :mod:`synth_datasets.cli`.

The ``generate`` function itself is torch- and fire-free and is tested directly. Only the argument parsing needs
``fire`` (the ``cli`` extra), which the torch-free CI leg does not install, so those tests skip without it.

"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from synth_datasets import cli
from synth_datasets.families.animals import AnimalShape


def _yaml_names(data_yaml: Path) -> list[str]:
    lines = data_yaml.read_text(encoding="utf-8").splitlines()
    return [line.split(": ", 1)[1] for line in lines[lines.index("names:") + 1 :]]


def test_missing_fire_points_at_the_cli_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "fire", None)

    with pytest.raises(SystemExit, match=r'pip install "vision-synth\[cli\]"'):
        cli.main(["generate", "out", "1"])


def test_generate_resolves_comma_separated_shape_names(tmp_path):
    counts = cli.generate(tmp_path, 3, fmt="yolo", img_size=32, seed=0, shapes="duck,camel")

    assert sum(counts.values()) == 3
    assert _yaml_names(tmp_path / "data.yaml") == [AnimalShape.DUCK.value, AnimalShape.CAMEL.value]


def test_generate_accepts_split_ratios_as_mapping_or_triple(tmp_path):
    assert cli.generate(tmp_path / "a", 4, img_size=32, seed=0, split_ratios={"train": 0.5, "holdout": 0.5}) == {
        "train": 2,
        "holdout": 2,
    }
    assert cli.generate(tmp_path / "b", 4, img_size=32, seed=0, split_ratios=(0.5, 0.5, 0.0)) == {"train": 2, "val": 2}


def test_main_generate_writes_a_dataset(tmp_path):
    pytest.importorskip("fire")
    out = tmp_path / "ds"
    argv = ["generate", str(out), "4", "--fmt", "yolo", "--task", "obb", "--img_size", "32", "--seed", "0"]

    cli.main([*argv, "--shapes", "duck,camel"])
    assert _yaml_names(out / "data.yaml") == ["duck", "camel"]

    cli.main([*argv, "--shapes", "duck", "--overwrite"])
    assert _yaml_names(out / "data.yaml") == ["duck"]


def test_main_reports_a_populated_output_dir_as_one_line_naming_the_flag(tmp_path, capsys):
    pytest.importorskip("fire")
    argv = ["generate", str(tmp_path / "ds"), "4", "--fmt", "yolo", "--img_size", "32", "--seed", "0"]
    cli.main(argv)
    capsys.readouterr()

    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)

    err = capsys.readouterr().err
    assert excinfo.value.code not in (0, None)
    assert "--overwrite" in err
    assert "overwrite=True" not in err
    assert "Traceback" not in err
    assert len(err.strip().splitlines()) == 1


def test_main_reports_an_invalid_value_as_one_line(tmp_path, capsys):
    pytest.importorskip("fire")

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["generate", str(tmp_path / "ds"), "4", "--img_size", "32", "--shapes", "unicorn"])

    err = capsys.readouterr().err
    assert excinfo.value.code not in (0, None)
    assert "unknown shape name 'unicorn'" in err
    assert "Traceback" not in err
    assert len(err.strip().splitlines()) == 1
    assert not (tmp_path / "ds").exists()


def test_main_accepts_colour_names(tmp_path):
    pytest.importorskip("fire")
    out = tmp_path / "ds"
    argv = ["generate", str(out), "4", "--fmt", "yolo", "--img_size", "32", "--seed", "0", "--class_mode", "color"]

    cli.main([*argv, "--colors", "red,blue"])
    assert _yaml_names(out / "data.yaml") == ["red", "blue"]
    cli.main([*argv, "--colors", "GREEN", "--overwrite"])
    assert _yaml_names(out / "data.yaml") == ["green"]


def test_module_entry_point_runs_in_a_fresh_interpreter(tmp_path):
    pytest.importorskip("fire")
    out = tmp_path / "ds"
    cmd = [sys.executable, "-m", "synth_datasets.cli", "generate", str(out), "2", "--img_size", "32", "--seed", "0"]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=120)  # noqa: S603

    assert result.returncode == 0, result.stderr
    assert (out / "train" / "_annotations.coco.json").is_file()


def test_console_script_points_at_main():
    tomllib = pytest.importorskip("tomllib")
    pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"

    scripts = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["scripts"]

    assert scripts["vision-synth"] == "synth_datasets.cli:main"
