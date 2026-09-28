"""Pin the one background that reads files the package does not ship.

Everything else here is procedural and answers only to its own parameters. `ImageBackground` reaches outside the
process, so it owes three extra things: it must refuse a directory with nothing usable in it rather than render black,
it must select the same file and the same crop for a given seed on any machine, and it must say which file a sample
stood on.

"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from synth_datasets.content.backgrounds import ImageBackground
from synth_datasets.core.config import SyntheticConfig
from synth_datasets.core.generator import SyntheticGenerator

IMG_SIZE = 32


@pytest.fixture
def picture_dir(tmp_path: Path) -> Path:
    """Return a directory of three distinguishable pictures, each larger than the canvas."""
    folder = tmp_path / "pictures"
    folder.mkdir()
    for index, level in enumerate((40, 120, 200)):
        canvas = np.full((64, 80, 3), level, dtype=np.uint8)
        canvas[:, :20] = 255 - level  # structure, so a crop is distinguishable from a flat fill
        Image.fromarray(canvas).save(folder / f"picture_{index}.png")
    return folder


def test_a_crop_is_the_canvas_size_and_dtype(picture_dir: Path) -> None:
    """A crop is handed to the rasterizer like any other canvas, so it owes the same shape contract."""
    canvas = ImageBackground(picture_dir).render(np.random.default_rng(0), IMG_SIZE)

    assert canvas.shape == (IMG_SIZE, IMG_SIZE, 3)
    assert canvas.dtype == np.uint8
    assert canvas.flags["C_CONTIGUOUS"]


def test_the_source_names_the_file_the_crop_came_from(picture_dir: Path) -> None:
    """Provenance is a path relative to the directory, in POSIX form so it reads the same on any host."""
    _canvas, source = ImageBackground(picture_dir).render_with_source(np.random.default_rng(0), IMG_SIZE)

    assert source in {"picture_0.png", "picture_1.png", "picture_2.png"}


def test_provenance_round_trips_to_the_sample(picture_dir: Path) -> None:
    """The chosen file reaches `sample.scene.background_source`, so a sample can be traced back.

    This is the field `SceneRecord` was given a second slot for; without it a photographic run would be unreproducible
    in practice even while being reproducible in principle.

    """
    config = SyntheticConfig(img_size=IMG_SIZE, background=ImageBackground(picture_dir))

    sample = next(iter(SyntheticGenerator(config).generate(1, seed=0)))

    assert sample.scene.background_source.endswith(".png")


def test_a_procedural_background_reports_no_source() -> None:
    """A mode with nothing outside the process to name leaves the field `None`."""
    sample = next(iter(SyntheticGenerator(SyntheticConfig(img_size=IMG_SIZE)).generate(1, seed=0)))

    assert sample.scene.background_source is None


def test_one_seed_selects_the_same_file_and_the_same_crop(picture_dir: Path) -> None:
    """Two renders from one seed agree on both choices, which is what makes the mode replayable.

    Files are listed in sorted order rather than in whatever order the filesystem returns, so the selection is a
    property of the seed rather than of the machine the dataset was built on.

    """
    background = ImageBackground(picture_dir)

    first = background.render_with_source(np.random.default_rng(4), IMG_SIZE)
    second = background.render_with_source(np.random.default_rng(4), IMG_SIZE)

    assert first[1] == second[1]
    assert np.array_equal(first[0], second[0])


def test_it_draws_exactly_the_file_index_and_two_crop_offsets(picture_dir: Path) -> None:
    """Three draws, always, so the count does not depend on how large the chosen file happens to be.

    Asserted against a twin stream running the documented calls, which is what stops the count being a comment rather
    than a contract.

    """
    background = ImageBackground(picture_dir)
    actual = np.random.default_rng(6)
    background.render(actual, IMG_SIZE)

    twin = np.random.default_rng(6)
    twin.integers(3)
    twin.integers(80 - IMG_SIZE + 1)
    twin.integers(64 - IMG_SIZE + 1)

    assert actual.random() == twin.random()


def test_grayscale_leaves_the_structure_and_drops_the_colour(tmp_path: Path) -> None:
    """A grayscale crop keeps its gradients and has identical channels, so colour classes stay unrivalled."""
    folder = tmp_path / "colourful"
    folder.mkdir()
    strip = np.zeros((40, 40, 3), dtype=np.uint8)
    strip[:, :20] = (200, 30, 30)
    strip[:, 20:] = (30, 30, 200)
    Image.fromarray(strip).save(folder / "strip.png")

    canvas = ImageBackground(folder, grayscale=True).render(np.random.default_rng(0), IMG_SIZE)

    assert np.array_equal(canvas[..., 0], canvas[..., 1])
    assert np.array_equal(canvas[..., 1], canvas[..., 2])
    assert canvas.min() < canvas.max()


def test_a_picture_smaller_than_the_canvas_is_scaled_up(tmp_path: Path) -> None:
    """A file with no crop to give is scaled proportionally rather than refused.

    Refusing would make the mode depend on the caller pre-sizing a whole directory, which is work the package can do
    once and correctly.

    """
    folder = tmp_path / "small"
    folder.mkdir()
    Image.fromarray(np.full((8, 12, 3), 77, dtype=np.uint8)).save(folder / "tiny.png")

    canvas = ImageBackground(folder).render(np.random.default_rng(0), IMG_SIZE)

    assert canvas.shape == (IMG_SIZE, IMG_SIZE, 3)


def test_a_missing_directory_is_refused_with_its_path_in_the_message(tmp_path: Path) -> None:
    """The mode fails at construction, not at the first rendered image, and says which path was wrong.

    The path is escaped because `match` is a regular expression, not a substring: a Windows `tmp_path` renders as
    `C:\\Users\\runneradmin\\...`, and `\\U` is an incomplete escape that fails the pattern before the call under test
    ever runs. On a POSIX runner the raw string happens to be a valid pattern, which is what let this reach CI.

    """
    missing = tmp_path / "nowhere"

    with pytest.raises(ValueError, match=re.escape(str(missing))):
        ImageBackground(missing)


def test_an_empty_directory_is_refused_with_its_path_in_the_message(tmp_path: Path) -> None:
    """An empty directory is the failure most worth refusing: it would otherwise render as nothing.

    Silently rendering black would produce a dataset that looks plausible in a thumbnail grid and carries none of the
    texture statistics the mode exists for.

    """
    folder = tmp_path / "empty"
    folder.mkdir()

    with pytest.raises(ValueError, match="holds no image"):
        ImageBackground(folder)


def test_a_directory_of_unreadable_suffixes_is_refused(tmp_path: Path) -> None:
    """Files that are not images do not count toward a directory being usable."""
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "readme.txt").write_text("not a picture", encoding="utf-8")

    with pytest.raises(ValueError, match="holds no image"):
        ImageBackground(folder)


def test_a_file_that_only_looks_like_an_image_is_refused(tmp_path: Path) -> None:
    """A corrupt file with an image suffix is caught at construction, not from inside rendering.

    Filtering on the filename alone made "readable" a promise about the suffix: the directory was accepted and the first
    generated image then raised `UnidentifiedImageError` from deep inside the renderer, far from the directory that
    caused it.

    """
    folder = tmp_path / "broken"
    folder.mkdir()
    (folder / "corrupt.png").write_bytes(b"this is not a png")

    with pytest.raises(ValueError, match="none could be parsed"):
        ImageBackground(folder)


def test_a_subdirectory_named_like_an_image_is_not_an_image(tmp_path: Path) -> None:
    """A directory called `shots.png` matched the suffix filter and counted as a usable image."""
    folder = tmp_path / "d"
    folder.mkdir()
    (folder / "shots.png").mkdir()

    with pytest.raises(ValueError, match="holds no image"):
        ImageBackground(folder)


def test_images_in_subdirectories_are_found_and_keep_their_path(tmp_path: Path) -> None:
    """The scan recurses, so provenance is a real relative path rather than always a bare filename.

    The field was converted to POSIX form for portability, which only means something once a path can contain a
    separator; before the scan recursed it never could.

    """
    folder = tmp_path / "shoot"
    (folder / "roll_one").mkdir(parents=True)
    Image.fromarray(np.full((40, 40, 3), 70, dtype=np.uint8)).save(folder / "roll_one" / "frame.png")

    _canvas, source = ImageBackground(folder).render_with_source(np.random.default_rng(0), IMG_SIZE)

    assert source == "roll_one/frame.png"


def test_the_mode_cannot_be_reached_without_a_directory() -> None:
    """`image_dir` has no default, so there is no way to ask for this mode without saying where from."""
    with pytest.raises(TypeError):
        ImageBackground()  # type: ignore[call-arg]


def test_a_photographic_canvas_leaves_every_placement_where_it_was(picture_dir: Path) -> None:
    """Like every other background, this one draws from the side stream and moves no object."""
    flat = SyntheticConfig(img_size=64, max_objects=3)
    photographic = SyntheticConfig(img_size=64, max_objects=3, background=ImageBackground(picture_dir))

    plain = [[a.bbox_xyxy for a in s.annotations] for s in SyntheticGenerator(flat).generate(3, seed=1)]
    over_photos = [[a.bbox_xyxy for a in s.annotations] for s in SyntheticGenerator(photographic).generate(3, seed=1)]

    assert plain == over_photos
