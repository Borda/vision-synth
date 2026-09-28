"""Shared SVG asset reading for every packaged shape family.

Three families ship artwork now — :mod:`~synth_datasets.families.animals` (traced silhouettes),
:mod:`~synth_datasets.families.symbols` (hand-authored outlines), and :mod:`~synth_datasets.families.letters` (stroke
graphs) — and all three are edited by the same tool, ``examples/edit_shape_keypoints.py``. One reader is what makes that
true: a document the loader accepts is read the same way by the editor, rather than through a second, stricter parser
that would silently misplace its vertices.

Two document shapes are supported, sharing the namespaces, the provenance block, the transform rejection, and the path
grammar:

**Outline documents** (animals, symbols) carry one closed ``<path id="outline">`` plus a ``<g id="keypoints">`` of named
``<circle>`` landmarks. The outline is the drawn silhouette; the landmarks annotate it.

**Graph documents** (letters) carry no outline at all. They carry ``<g id="nodes">`` of named ``<circle>`` grid
positions and ``<g id="strokes">`` of ``<line>`` edges, optionally curved by a ``zoo:bulge`` attribute and cut by
``zoo:cut``. The silhouette is *generated* from that graph at load time by stroking it, so what the asset stores is the
letter's skeleton rather than its outline.

Only straight-line path commands are accepted (``M``/``L``/``H``/``V``/``Z``, absolute or relative). A curve command is
rejected with an authoring hint rather than silently flattened: Inkscape writes curves whenever a node is smoothed, and
a flattened approximation would drift from what the author saw.

"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from importlib.abc import Traversable

#: Provenance attributes every document must carry (as ``zoo:``-namespaced root attributes).
#: ``attribution`` is deliberately excluded — CC0/PDM art carries no attribution obligation.
REQUIRED_PROVENANCE: tuple[str, ...] = ("origin", "title", "license", "note")

#: Every provenance key read off a document, in the order they are reported.
PROVENANCE_KEYS: tuple[str, ...] = ("origin", "title", "license", "attribution", "note")

#: SVG commands accepted by :func:`parse_path_d`, absolute and relative.
_SUPPORTED_PATH_COMMANDS = "MmLlHhVvZz"

#: Curve commands rejected with an authoring hint — anything with an SVG letter that isn't a straight
#: line, a move, or a close belongs to this set.
_CURVE_PATH_COMMANDS = "CcSsQqTtAa"

_SVG_NS = "http://www.w3.org/2000/svg"
_ZOO_NS = "https://github.com/Borda/vision-synth/ns/zoo"


def svg_tag(tag: str) -> str:
    """Return ``tag`` fully qualified with the SVG namespace, for ``ElementTree`` lookups."""
    return f"{{{_SVG_NS}}}{tag}"


def zoo_attr(name: str) -> str:
    """Return ``name`` fully qualified with the ``zoo:`` namespace, for ``ElementTree`` lookups."""
    return f"{{{_ZOO_NS}}}{name}"


def parse_path_d(d: str, name: str) -> list[tuple[float, float]]:
    """Parse an SVG path ``d`` attribute into a closed polygon's vertex list.

    Tolerant of both absolute and relative ``M``/``L``/``H``/``V``/``Z`` and of SVG's implicit
    command repetition (a bare coordinate pair following ``M``/``L`` repeats the last command) —
    that tolerance is what lets a re-saved Inkscape path (relative, ``H``/``V``-heavy) still parse,
    rather than only the absolute ``M``/``L``/``Z`` this package's own authoring script emits.

    Args:
        d: The ``<path>`` element's ``d`` attribute.
        name: Animal name, for error messages.

    Returns:
        The outline vertices in path order.

    Raises:
        ValueError: If the path uses a curve command, an unrecognised command, a second subpath
            (a second ``M``/``m``), a moveto without a coordinate pair, ends part-way through a
            coordinate pair, puts a command letter where the second half of a pair belongs, carries
            coordinates after its closing ``Z``/``z``, or is not closed with a trailing ``Z``/``z``.

    """
    raw_tokens = re.findall(r"[A-Za-z]|-?\d*\.?\d+(?:[eE][-+]?\d+)?", d)
    points: list[tuple[float, float]] = []
    current = (0.0, 0.0)
    cmd: str | None = None
    started = False
    closed = False
    i = 0
    while i < len(raw_tokens):
        tok = raw_tokens[i]
        if tok[:1].isalpha():
            # A moveto that consumed a pair leaves ``cmd`` as the implicit lineto it decays to, so
            # ``cmd`` still being ``M``/``m`` here means the moveto never got its coordinates — the
            # first vertex would otherwise be dropped without a word.
            if cmd in {"M", "m"}:
                raise ValueError(f"document {name}.svg has a moveto without a coordinate pair")
            cmd, started, closed, i = _consume_command(tok, name, started, i)
            continue
        if cmd is None:
            raise ValueError(f"document {name}.svg path data must start with a moveto command")
        if cmd in "Zz":
            # ``Z``/``z`` takes no arguments, so a numeric token here is malformed path data. Without
            # this guard it fell through every branch of _consume_coordinate to the relative-lineto
            # default and was silently absorbed as a phantom vertex.
            raise ValueError(
                f"document {name}.svg has coordinates after the closing Z; a close command takes no arguments"
            )
        current, cmd, consumed = _consume_coordinate(cmd, name, raw_tokens, i, current)
        points.append(current)
        i += consumed
    if not closed:
        raise ValueError(f"document {name}.svg path is not closed; append a trailing Z")
    return points


def _consume_command(tok: str, name: str, started: bool, i: int) -> tuple[str, bool, bool, int]:
    """Validate one command letter token and return ``(cmd, started, closed, next_index)``."""
    if tok in _CURVE_PATH_COMMANDS:
        raise ValueError(
            f"document {name}.svg uses a curve command {tok!r}; in Inkscape use Path ▸ Flatten "
            "to straight segments before saving"
        )
    if tok not in _SUPPORTED_PATH_COMMANDS:
        raise ValueError(f"document {name}.svg path data has an unsupported command {tok!r}")
    if tok in "Mm" and started:
        raise ValueError(f"document {name}.svg path has a second subpath (a second M/m); expected exactly one")
    if tok in "Zz":
        return tok, started, True, i + 1
    return tok, started or tok in "Mm", False, i + 1


def _consume_coordinate(
    cmd: str, name: str, tokens: list[str], i: int, current: tuple[float, float]
) -> tuple[tuple[float, float], str, int]:
    """Apply one numeric argument (or coordinate pair) to ``current``; returns the new command.

    A bare coordinate pair following ``M``/``m`` is an implicit lineto for every pair after the first, per the SVG path
    grammar — the returned ``cmd`` reflects that so the next bare pair (if any) is interpreted correctly too.

    ``name`` is threaded in for the same reason :func:`_consume_command` takes it: a two-token read has to be validated
    before it happens, and the resulting error belongs to the named-``ValueError`` contract :func:`parse_path_d`
    documents — not to a raw ``IndexError`` off the end of the token list, nor to :func:`float`'s own message about a
    command letter, neither of which names the document that failed to load.

    Args:
        cmd: The command in effect for this argument, absolute or relative.
        name: Animal name, for error messages.
        tokens: The whole token list, so a coordinate *pair* can be read (and bounds-checked) as one unit.
        i: Index of the first argument token to consume.
        current: The point the path is standing on, which relative commands offset from.

    Returns:
        The new current point, the command in effect for the next argument, and how many tokens were consumed.

    Raises:
        ValueError: If a pair is cut short by the end of the path data, or if its second half is a command letter.

    """
    if cmd in "Hh":
        x = float(tokens[i])
        return (current[0] + x if cmd == "h" else x, current[1]), cmd, 1
    if cmd in "Vv":
        y = float(tokens[i])
        return (current[0], current[1] + y if cmd == "v" else y), cmd, 1
    # Every command left takes an (x, y) pair, so the second token must exist and must be a number. The tokenizer emits
    # only command letters and numerals, which is what makes a leading alpha the exact test for "this is not a
    # coordinate".
    if i + 1 >= len(tokens):
        raise ValueError(f"document {name}.svg path data ends mid-coordinate-pair after {tokens[i]!r}")
    if tokens[i + 1][:1].isalpha():
        raise ValueError(
            f"document {name}.svg has the command {tokens[i + 1]!r} inside a coordinate pair, "
            f"where the second half of the pair opened by {tokens[i]!r} belongs"
        )
    x, y = float(tokens[i]), float(tokens[i + 1])
    if cmd == "M":
        return (x, y), "L", 2
    if cmd == "m":
        return (current[0] + x, current[1] + y), "l", 2
    if cmd == "L":
        return (x, y), "L", 2
    return (current[0] + x, current[1] + y), "l", 2  # cmd == "l"


def reject_transforms(root: ET.Element, name: str) -> None:
    """Raise if any element in the document carries a ``transform`` attribute.

    A transform would place the drawn geometry somewhere other than where its coordinates say, so every reader here
    would see the untransformed points while an author sees the transformed ones.

    """
    for element in root.iter():
        if "transform" in element.attrib:
            tag = element.tag.rsplit("}", 1)[-1]
            raise ValueError(
                f"document {name}.svg has a transform on <{tag}>; in Inkscape set "
                "Preferences ▸ Behavior ▸ Transforms ▸ Store transformation = Optimized"
            )


def read_provenance(root: ET.Element, name: str) -> dict[str, str]:
    """Return the document's ``zoo:`` provenance attributes, requiring the mandatory ones.

    Args:
        root: The parsed document root.
        name: The document stem, for error messages.

    Returns:
        The provenance keys present on the root, in :data:`PROVENANCE_KEYS` order.

    Raises:
        ValueError: If any of :data:`REQUIRED_PROVENANCE` is absent or empty.

    """
    missing = [key for key in REQUIRED_PROVENANCE if not root.get(zoo_attr(key))]
    if missing:
        raise ValueError(f"document {name}.svg is missing the key(s) {missing}")
    return {key: value for key in PROVENANCE_KEYS if (value := root.get(zoo_attr(key)))}


def read_named_circles(
    root: ET.Element, name: str, group_id: str, allowed: Sequence[str], required: Iterable[str] = ()
) -> dict[str, tuple[float, float]]:
    """Parse a ``<g id=...>`` group of ``zoo:name``-tagged ``<circle>`` elements into a point map.

    Shared by the landmark group of an outline document and the node group of a graph document —
    the two differ only in which names they allow and which of those are mandatory.

    Args:
        root: The parsed document root.
        name: The document stem, for error messages.
        group_id: The ``id`` of the group to read (``"keypoints"`` or ``"nodes"``).
        allowed: Every name a circle in this group may carry.
        required: The subset that must be present. Anything in ``allowed`` but not here is optional
            and simply absent from the result.

    Returns:
        ``name -> (cx, cy)`` for every circle present.

    Raises:
        ValueError: If the group is missing, a circle carries an unknown or duplicate ``zoo:name``,
            a circle has no ``cx``/``cy``, or a required name is absent.

    """
    group = root.find(f"{svg_tag('g')}[@id='{group_id}']")
    if group is None:
        raise ValueError(f'document {name}.svg is missing the <g id="{group_id}"> group')
    seen: dict[str, tuple[float, float]] = {}
    for circle in group.findall(svg_tag("circle")):
        point_name = circle.get(zoo_attr("name"))
        if point_name is None or point_name not in allowed:
            raise ValueError(
                f"document {name}.svg has an unknown or missing zoo:name {point_name!r}; "
                f"expected one of {tuple(allowed)}"
            )
        if point_name in seen:
            raise ValueError(f"document {name}.svg has a duplicate zoo:name {point_name!r}")
        # A present-but-coordinate-less circle is malformed whether the point is mandatory or
        # optional, so it is rejected here rather than defaulted to NaN: the mandatory-point guard
        # below only tests key presence, so a NaN sentinel would sail straight through it.
        cx, cy = circle.get("cx"), circle.get("cy")
        if cx is None or cy is None:
            raise ValueError(f"document {name}.svg point {point_name!r} is missing cx/cy")
        seen[point_name] = (float(cx), float(cy))
    absent = [key for key in required if key not in seen]
    if absent:
        raise ValueError(f"document {name}.svg is missing the point(s) {absent}")
    return seen


def _parse_root(asset: Traversable, name: str) -> ET.Element:
    """Parse one packaged document, rejecting transforms before anything reads its geometry."""
    # ``str()`` on a Traversable is not guaranteed to yield an openable path (a zipimport or frozen
    # loader has no real file behind it), so the document is read through the resource's own opener.
    # Binary, not text: that leaves any XML declaration's encoding for ElementTree to honor.
    with (asset / f"{name}.svg").open("rb") as handle:
        root = ET.parse(handle).getroot()  # noqa: S314 - packaged asset addressed by a fixed enum name, never untrusted input
    reject_transforms(root, name)
    return root


def read_outline_document(
    asset: Traversable, name: str, keypoint_names: Sequence[str], required_keypoints: Iterable[str]
) -> tuple[list[tuple[float, float]], dict[str, tuple[float, float]], dict[str, str]]:
    """Read one outline document: a closed silhouette plus its named landmarks.

    Args:
        asset: The packaged directory holding ``<name>.svg``.
        name: The document stem.
        keypoint_names: Every landmark name the family defines.
        required_keypoints: The landmarks that must be present; the rest are optional.

    Returns:
        The outline vertices, the landmarks present (by name), and the provenance attributes.

    Raises:
        ValueError: If the document uses a curve or a transform, does not have exactly one closed
            simple path, or is missing a mandatory landmark, a landmark's coordinates, or a
            provenance attribute. A malformed document is rejected here, at import, rather than
            surfacing mid-generation as a lookup failure.

    """
    root = _parse_root(asset, name)
    paths = root.findall(svg_tag("path"))
    if len(paths) != 1:
        raise ValueError(f"document {name}.svg must have exactly one <path>, found {len(paths)}")
    outline = parse_path_d(paths[0].get("d", ""), name)
    provenance = read_provenance(root, name)
    keypoints = read_named_circles(root, name, "keypoints", keypoint_names, required_keypoints)
    return outline, keypoints, provenance


def read_graph_document(
    asset: Traversable, name: str, node_names: Sequence[str]
) -> tuple[dict[str, tuple[float, float]], list[tuple[str, str, float]], list[tuple[str, str, float]], dict[str, str]]:
    """Read one graph document: named nodes plus the stroke edges connecting them.

    A graph document has no outline. The silhouette is generated by stroking these edges at load
    time, which is why the asset stores the skeleton: stroke width and counter gap stay tunable, and
    a node position remains a single number an editor can drag rather than a consequence baked into
    hundreds of outline vertices.

    Args:
        asset: The packaged directory holding ``<name>.svg``.
        name: The document stem.
        node_names: Every node name the family defines. All are optional — a letter touches only the
            grid slots its strokes need.

    Returns:
        The nodes present (by name), the stroke edges as ``(start, end, bulge)``, the counter cuts as
        ``(start, end, position)``, and the provenance attributes. ``bulge`` is ``0.0`` for a
        straight edge.

    Raises:
        ValueError: If the document uses a transform, an edge names a node the document does not
            define, an edge is missing an endpoint, or a provenance attribute is absent.

    """
    root = _parse_root(asset, name)
    provenance = read_provenance(root, name)
    nodes = read_named_circles(root, name, "nodes", node_names)
    edges = _read_edges(root, name, "strokes", nodes, "bulge")
    cuts = _read_edges(root, name, "cuts", nodes, "cut", optional_group=True)
    return nodes, edges, cuts, provenance


def _read_edges(
    root: ET.Element,
    name: str,
    group_id: str,
    nodes: dict[str, tuple[float, float]],
    value_attr: str,
    optional_group: bool = False,
) -> list[tuple[str, str, float]]:
    """Parse a ``<g id=...>`` group of ``<line>`` elements naming node endpoints.

    Args:
        root: The parsed document root.
        name: The document stem, for error messages.
        group_id: The ``id`` of the group to read (``"strokes"`` or ``"cuts"``).
        nodes: The node map the endpoints must resolve against.
        value_attr: The ``zoo:`` attribute carrying each edge's scalar (``"bulge"`` or ``"cut"``).
        optional_group: When true, a missing group yields no edges instead of raising — a letter
            with no counters has no cuts group at all.

    Returns:
        ``(start, end, value)`` per line, in document order. ``value`` is ``0.0`` when the attribute
        is absent.

    Raises:
        ValueError: If a required group is missing, or an edge names an endpoint the document's node
            group does not define.

    """
    group = root.find(f"{svg_tag('g')}[@id='{group_id}']")
    if group is None:
        if optional_group:
            return []
        raise ValueError(f'document {name}.svg is missing the <g id="{group_id}"> group')
    edges: list[tuple[str, str, float]] = []
    for line in group.findall(svg_tag("line")):
        start, end = line.get(zoo_attr("start")), line.get(zoo_attr("end"))
        if start is None or end is None:
            raise ValueError(f"document {name}.svg has a <line> in {group_id!r} missing zoo:start/zoo:end")
        unknown = [point for point in (start, end) if point not in nodes]
        if unknown:
            raise ValueError(
                f"document {name}.svg edge {start!r}->{end!r} names undefined node(s) {unknown}; "
                f"every endpoint must have a <circle> in the nodes group"
            )
        raw = line.get(zoo_attr(value_attr))
        edges.append((start, end, float(raw) if raw is not None else 0.0))
    return edges
