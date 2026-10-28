import json
import shutil

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

ORIGIN_X, ORIGIN_Y = 1_000_000.0, 2_000_000.0  # the top-left corner of every synthetic chip, in EPSG:3857


@pytest.fixture
def write_chip(tmp_path):
    """Write a synthetic georeferenced RGB GeoTIFF (EPSG:3857, 1 m pixels)."""

    def _write(name: str = "chip.tif", height: int = 256, width: int = 256, count: int = 3):
        path = tmp_path / name
        data = np.random.default_rng(0).integers(0, 256, size=(count, height, width), dtype=np.uint8)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=count,
            dtype="uint8",
            crs="EPSG:3857",
            transform=from_origin(ORIGIN_X, ORIGIN_Y, 1.0, 1.0),
        ) as dst:
            dst.write(data)
        return path

    return _write


# A tile near the equator at zoom 20: 14.9 cm/px, so a 256 px chip is scaled by about 2.99.
CHIP_NAME = "OAM-524288-524288-20.tif"


def lon_lat(col: float, row: float) -> list[float]:
    """The longitude and latitude of chip pixel `(col, row)` of a synthetic chip (1 m pixels, origin above)."""
    return list(
        Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform(
            ORIGIN_X + col, ORIGIN_Y - row
        )
    )


def tree_point(col: float, row: float, **tags: str) -> dict:
    return {
        "type": "Feature",
        "properties": {"tags": {"natural": "tree", **tags}},
        "geometry": {"type": "Point", "coordinates": lon_lat(col, row)},
    }


@pytest.fixture
def labelled_chips(write_chip, tmp_path):
    """Four chips of one zoom-20 block, all with the same georeferencing, and one GeoJSON of labels in EPSG:4326.

    The labels: a polygon over the top-left 128 x 128 px, a tree point with a 10 m crown at pixel (200, 60), and a tree
    point without a diameter at (60, 210). The chip name says 0.149 m/px, so the disc of 10 m has a radius of 33.5 px.
    """
    chips, labels = tmp_path / "chips", tmp_path / "labels"
    chips.mkdir()
    labels.mkdir()
    names = [f"OAM-{524288 + index}-524288-20.tif" for index in range(4)]
    for name in names:
        shutil.copy(write_chip(name=name), chips / name)
    square = [lon_lat(0, 0), lon_lat(128, 0), lon_lat(128, 128), lon_lat(0, 128), lon_lat(0, 0)]
    features = [
        {"type": "Feature", "properties": {"natural": "tree_group"}, "geometry": {"type": "Polygon", "coordinates": [square]}},
        tree_point(200, 60, diameter_crown="10"),
        tree_point(60, 210),
    ]  # fmt: skip
    (labels / "labels.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return chips, labels, names
