"""Dataset helpers for ts-hot (numpy, rasterio and pyproj only, no torch; safe to import at serving time).

- `list_chip_paths`, `spatial_split`: find the chips and split them into train and validation.
- `build_chip_labels`: one GeoJSON of polygon and tree-point labels -> per-chip label shapes in chip pixels, nothing
  copied. `load_labels` takes either that mapping or the GeoJSON path.
- `scale_chip_and_labels`, `augment_chip_and_mask`: the per-chip steps of training, in the order they run. The first
  rasterises the labels at the model's resolution; the second acts on image and mask together.

A mask holds `BACKGROUND`, `TREE` and `IGNORE` (pixels left out of the loss and of the scores). A label shape is a dict
`{"kind": "tree" | "ignore", "rings": [(N, 2) array of (col, row) vertices, ...]}`, the exterior ring first.
"""

import json
import math
import re
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import Transformer
from rasterio.features import rasterize

from .params import PreprocessParams
from .preprocess import preprocess_rgb
from .tiles import OAM_TILE_RE, chip_scale, tile_gsd_m

CHIP_PATTERNS = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")
LABEL_SUFFIXES = (".geojson", ".json")
BACKGROUND, TREE, IGNORE = 0, 1, 255
DEFAULT_CROWN_M = 5.0  # diameter of the ignored disc of a tree point without a usable `diameter_crown`
MAX_CROWN_M = 50.0  # the largest crown a point may have; chips look this far beyond their edge for points
METRES_PER_DEGREE = 111_320  # spherical approximation, within 1% of WGS84 latitude degrees
CIRCLE_VERTICES = 32
DIAMETER_PATTERN = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*(?:m|meter|meters|metre|metres)?\s*$", re.IGNORECASE)


def list_chip_paths(root: str | Path) -> list[Path]:
    """RGB chips in a folder. Training and evaluation read any of them (the scale comes from the OAM name), but turning
    GeoJSON labels into chip pixels needs georeferenced GeoTIFFs."""
    folder = Path(root)
    return sorted(path for pattern in CHIP_PATTERNS for path in folder.glob(pattern) if path.is_file())


def spatial_split(
    chip_names: list[str], val_ratio: float, seed: int, *, block_size: int = 4
) -> tuple[list[str], list[str]]:
    """Block-spatial split on OAM tile coords; whole `(x // K, y // K)` blocks go to one side.

    Neighbouring chips share trees cut at chip borders and similar canopy, so a random
    chip split leaks. Every name must be `OAM-x-y-z.ext`; anything else raises `ValueError`.
    Returns sorted `(train, val)` name lists.
    """
    blocks_map: dict[tuple[int, int], list[str]] = {}
    for name in chip_names:
        match = OAM_TILE_RE.match(name)
        if match is None:
            raise ValueError(f"chip name {name!r} does not match OAM-<x>-<y>-<z>.<ext>")
        blocks_map.setdefault(
            (int(match.group(1)) // block_size, int(match.group(2)) // block_size), []
        ).append(name)

    rng = np.random.default_rng(seed)
    blocks = sorted(blocks_map)
    rng.shuffle(blocks)

    n_val_target = max(1, int(len(chip_names) * val_ratio)) if chip_names else 0

    val: list[str] = []
    train: list[str] = []
    for block in blocks:
        (val if len(val) < n_val_target else train).extend(blocks_map[block])
    return sorted(train), sorted(val)


def _find_labels_geojson(labels: str | Path) -> Path:
    """The labels file: the path itself if it is a `.geojson` / `.json` file, else the one such file under a folder."""
    root = Path(labels)
    if root.is_file() and root.suffix.lower() in LABEL_SUFFIXES:
        return root
    matches = sorted(root.rglob("*.geojson")) or sorted(root.rglob("*.json"))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one .geojson (or .json) in {root}, found {len(matches)}")
    return matches[0]


def _parse_geojson_crs(feature_collection: dict[str, Any]) -> str:
    """CRS name from the legacy GeoJSON `crs` member; RFC 7946 files default to EPSG:4326."""
    crs = feature_collection.get("crs")
    if isinstance(crs, dict):
        props = crs.get("properties")
        if isinstance(props, dict) and isinstance(props.get("name"), str) and props["name"]:
            return props["name"]
    return "EPSG:4326"


def parse_diameter_m(raw: Any) -> float | None:
    """A `diameter_crown` tag in metres: `4`, `4.5`, `4m`, `4 m`, `4 meters` and a comma as decimal (`3,5 m`) are read.

    Anything else, a missing value, or a value that is not above zero gives `None`.
    """
    if raw is None:
        return None
    match = DIAMETER_PATTERN.match(str(raw))
    if match is None:
        return None
    diameter = float(match.group(1).replace(",", "."))
    return diameter if math.isfinite(diameter) and diameter > 0 else None


def _polygon_shape(rings: list[Any]) -> dict[str, Any]:
    arrays = [np.asarray(ring, dtype=float)[:, :2] for ring in rings]
    exterior = arrays[0]
    return {"kind": "polygon", "rings": arrays, "bounds": (*exterior.min(axis=0), *exterior.max(axis=0))}


def read_label_shapes(labels_geojson: str | Path) -> tuple[list[dict[str, Any]], str]:
    """Polygons and tree points of a GeoJSON as shapes in the file's CRS, plus that CRS.

    A polygon feature gives one polygon shape (a multipolygon one per part); a point feature gives a point shape with
    its `diameter_crown` in metres (`None` if it has no usable one), read from `properties.tags` or from the
    properties. A point is taken as longitude and latitude in degrees. Other geometries, such as lines, are skipped.
    """
    collection = json.loads(Path(labels_geojson).read_text(encoding="utf-8"))
    if not isinstance(collection, dict) or collection.get("type") != "FeatureCollection":
        raise ValueError("labels file must be a GeoJSON FeatureCollection")
    features = collection.get("features", [])
    if not isinstance(features, list):
        raise ValueError("GeoJSON FeatureCollection 'features' must be a list")
    shapes: list[dict[str, Any]] = []
    for feature in features:
        geometry = feature.get("geometry") if isinstance(feature, dict) else None
        if not isinstance(geometry, dict):
            continue
        kind, coordinates = geometry.get("type"), geometry.get("coordinates")
        if kind == "Polygon" and coordinates:
            shapes.append(_polygon_shape(coordinates))
        elif kind == "MultiPolygon":
            shapes.extend(_polygon_shape(part) for part in coordinates if part)
        elif kind == "Point" and coordinates:
            properties = feature.get("properties") or {}
            tags = properties.get("tags") or properties
            x, y = float(coordinates[0]), float(coordinates[1])
            shapes.append(
                {"kind": "point", "x": x, "y": y, "diameter_m": parse_diameter_m(tags.get("diameter_crown")),
                 "bounds": (x, y, x, y)}
            )  # fmt: skip
    return shapes, _parse_geojson_crs(collection)


@lru_cache(maxsize=16)
def _transformer(source_crs: str, target_crs: str) -> Transformer:
    """A coordinate transformer, made once: building one costs far more than using it on a chip's few vertices."""
    return Transformer.from_crs(source_crs, target_crs, always_xy=True)


def _circle_ring(col: float, row: float, radius: float) -> np.ndarray:
    """A closed ring of `CIRCLE_VERTICES` vertices around `(col, row)`."""
    angles = np.linspace(0.0, 2.0 * np.pi, CIRCLE_VERTICES, endpoint=False)
    ring = np.stack([col + radius * np.cos(angles), row + radius * np.sin(angles)], axis=1)
    return np.vstack([ring, ring[:1]])


def chip_shapes(
    chip_path: str | Path,
    *,
    shapes: list[dict[str, Any]],
    bounds: np.ndarray,
    labels_crs: str,
    default_crown_m: float = DEFAULT_CROWN_M,
) -> list[dict[str, Any]]:
    """The label shapes that touch a chip, in chip pixels.

    A polygon becomes a `tree` shape. A point becomes a disc of its `diameter_crown` metres at the chip's true ground
    resolution (a `tree` shape), or, with no usable diameter, a disc of `default_crown_m` that is `ignore`: the model is
    neither rewarded nor punished there. A crown that crosses the chip's edge is drawn from the points of the whole file.
    """
    import rasterio

    with rasterio.open(chip_path) as src:
        width, height, transform, crs = int(src.width), int(src.height), src.transform, src.crs
    if crs is None:
        raise ValueError(f"{chip_path} has no CRS; GeoJSON labels need a georeferenced GeoTIFF chip")
    gsd = tile_gsd_m(Path(chip_path).name, width)
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=float)
    corner_x = transform.a * corners[:, 0] + transform.b * corners[:, 1] + transform.c
    corner_y = transform.d * corners[:, 0] + transform.e * corners[:, 1] + transform.f
    same_crs = str(crs) == str(labels_crs)
    to_chip = None if same_crs else _transformer(str(labels_crs), str(crs))
    if not same_crs:
        corner_x, corner_y = _transformer(str(crs), str(labels_crs)).transform(corner_x, corner_y)
    margin = (
        MAX_CROWN_M / METRES_PER_DEGREE
    )  # points are in degrees: a crown reaches this far past the chip edge
    near = (
        (bounds[:, 2] >= min(corner_x) - margin) & (bounds[:, 0] <= max(corner_x) + margin)
        & (bounds[:, 3] >= min(corner_y) - margin) & (bounds[:, 1] <= max(corner_y) + margin)
    )  # fmt: skip
    inverse = ~transform

    def to_pixels(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if to_chip is not None:
            x, y = to_chip.transform(x, y)
        # Explicit affine maths (same reason as georef.py): works across `affine` versions.
        return np.stack(
            [inverse.a * x + inverse.b * y + inverse.c, inverse.d * x + inverse.e * y + inverse.f], axis=1
        )

    found: list[dict[str, Any]] = []
    for index in np.flatnonzero(near):
        shape = shapes[index]
        if shape["kind"] == "polygon":
            rings = [to_pixels(ring[:, 0], ring[:, 1]) for ring in shape["rings"]]
            found.append({"kind": "tree", "rings": rings})
        else:
            col, row = to_pixels(np.array([shape["x"]]), np.array([shape["y"]]))[0]
            usable = shape["diameter_m"] is not None
            radius = (shape["diameter_m"] if usable else default_crown_m) / 2.0 / gsd
            found.append({"kind": "tree" if usable else "ignore", "rings": [_circle_ring(col, row, radius)]})
    return [
        shape
        for shape in found
        if shape["rings"][0][:, 0].max() > 0 and shape["rings"][0][:, 0].min() < width
        and shape["rings"][0][:, 1].max() > 0 and shape["rings"][0][:, 1].min() < height
    ]  # fmt: skip


def build_chip_labels(
    chips_dir: str | Path, labels: str | Path, *, default_crown_m: float = DEFAULT_CROWN_M
) -> dict[str, list[dict[str, Any]]]:
    """Label shapes of every chip in `chips_dir` from one GeoJSON of polygon and tree-point labels. Nothing is copied.

    `labels` is the `.geojson` / `.json` file, or a folder holding exactly one. A pipeline step can call this on the
    chips and labels it was given, so no prepared folder has to be shared between isolated steps. Chips without a
    label get an empty list (all background).
    """
    shapes, crs = read_label_shapes(_find_labels_geojson(labels))
    bounds = np.array([shape["bounds"] for shape in shapes], dtype=float).reshape(-1, 4)
    return {
        path.name: chip_shapes(
            path, shapes=shapes, bounds=bounds, labels_crs=crs, default_crown_m=default_crown_m
        )
        for path in list_chip_paths(chips_dir)
    }


def load_labels(
    chips_dir: str | Path, labels: str | Path | Mapping[str, list[dict[str, Any]]]
) -> Mapping[str, list[dict[str, Any]]]:
    """The per-chip label shapes themselves, or built from a labels GeoJSON path (`build_chip_labels`)."""
    if isinstance(labels, Mapping):
        return labels
    return build_chip_labels(chips_dir, labels)


def rasterize_chip_labels(shapes: list[dict[str, Any]], height: int, width: int) -> np.ndarray:
    """`(H, W)` `uint8` mask of label shapes in pixels: `IGNORE` discs first, then `TREE` over them, background elsewhere."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for kind, value in (("ignore", IGNORE), ("tree", TREE)):
        geometries = [
            {"type": "Polygon", "coordinates": [ring.tolist() for ring in shape["rings"]]}
            for shape in shapes
            if shape["kind"] == kind
        ]
        if geometries:
            rasterize([(geometry, value) for geometry in geometries], out=mask, default_value=value)
    return mask


def scale_chip_and_labels(
    rgb: np.ndarray, shapes: list[dict[str, Any]], pre: PreprocessParams, name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Scale an RGB chip exactly as at serving time and draw its labels at that resolution.

    The label shapes (chip pixels) are scaled with the chip and rasterised at the scaled size, so the mask edges are
    as sharp as the image. Returns the `(3, H', W')` float32 image and the `(H', W')` `uint8` mask.
    """
    scale = chip_scale(
        name, rgb.shape[1], pre.model_gsd_m, height_px=rgb.shape[0], max_input_px=pre.max_input_px
    )
    image, meta = preprocess_rgb(rgb, scale)
    factors = np.array([meta.scale_x, meta.scale_y])
    scaled = [{**shape, "rings": [ring * factors for ring in shape["rings"]]} for shape in shapes]
    return image, rasterize_chip_labels(scaled, image.shape[1], image.shape[2])


def augment_chip_and_mask(
    image: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """A random quarter turn and a random mirror, the same for the image and the mask (aerial imagery has no up)."""
    turns = int(rng.integers(0, 4))
    mirror = bool(rng.integers(0, 2))
    image, mask = np.rot90(image, turns, axes=(1, 2)), np.rot90(mask, turns, axes=(0, 1))
    if mirror:
        image, mask = image[:, :, ::-1], mask[:, ::-1]
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)
