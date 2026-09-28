"""A split name must be one plain path component, since each becomes a directory under ``output_dir``.

``SplitRatios.custom({"../escaped": 1.0})`` used to be accepted, and a COCO export then created ``escaped/`` *beside*
``output_dir`` rather than inside it (YOLO nests one level deeper, so ``../../x`` escaped there).

"""

from __future__ import annotations

import pytest

from synth_datasets import SplitRatios, generate_dataset


@pytest.mark.parametrize("name", ["../escaped", "../../x", "a/b", "a\\b", "/abs", "", ".", "..", "nul\x00", 1])
def test_custom_split_names_must_be_one_safe_path_component(name):
    with pytest.raises(ValueError, match="split name"):
        SplitRatios.custom({name: 1.0})


def test_split_names_are_checked_on_direct_construction_too():
    with pytest.raises(ValueError, match="split name"):
        SplitRatios(named={"../x": 1.0})


def test_ordinary_custom_split_names_pass():
    ratios = SplitRatios.custom({"train": 0.5, "calib-2": 0.25, "hold.out": 0.25})
    assert ratios.to_dict() == {"train": 0.5, "calib-2": 0.25, "hold.out": 0.25}


def test_escaping_split_never_reaches_the_disk(tmp_path):
    with pytest.raises(ValueError, match="split name"):
        generate_dataset(tmp_path / "ds", 2, fmt="coco", img_size=32, split_ratios=SplitRatios.custom({"../x": 1.0}))
    assert not any(tmp_path.iterdir())
