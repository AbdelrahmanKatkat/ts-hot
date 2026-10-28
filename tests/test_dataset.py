import json

import numpy as np
import pytest

from conftest import CHIP_NAME, lon_lat, tree_point
from ts_hot.dataset import (
    BACKGROUND,
    IGNORE,
    TREE,
    augment_chip_and_mask,
    build_chip_labels,
    load_labels,
    parse_diameter_m,
    rasterize_chip_labels,
    read_label_shapes,
    scale_chip_and_labels,
    spatial_split,
)
from ts_hot.params import PreprocessParams

GSD = 0.1493  # metres per pixel of a zoom-20 tile near the equator, from the chip name


def _names(n_x: int, n_y: int):
    return [f"OAM-{x}-{y}-19.tif" for x in range(n_x) for y in range(n_y)]


def test_split_is_deterministic_and_disjoint():
    names = _names(16, 16)
    a = spatial_split(names, 0.2, 1)
    assert a == spatial_split(names, 0.2, 1) and a != spatial_split(names, 0.2, 2)
    train, val = a
    assert set(train).isdisjoint(val) and sorted(train + val) == sorted(names)


def test_whole_blocks_stay_together():
    train, val = spatial_split(_names(16, 16), 0.25, 0, block_size=4)
    block = lambda n: tuple(int(v) // 4 for v in n.split("-")[1:3])  # noqa: E731
    assert {block(n) for n in train}.isdisjoint({block(n) for n in val})


def test_non_oam_name_raises_and_empty_input():
    with pytest.raises(ValueError, match="OAM"):
        spatial_split(["OAM-1-2-3.tif", "a.tif"], 0.4, 0)
    assert spatial_split([], 0.2, 0) == ([], [])


@pytest.mark.parametrize(
    "raw, expected",
    [("4", 4.0), (4, 4.0), ("4.5", 4.5), ("4m", 4.0), ("4 m", 4.0), ("3,5 m", 3.5), (" 12 meters ", 12.0)],
)
def test_the_diameter_formats_of_the_tag_are_read(raw, expected):
    assert parse_diameter_m(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "3 m wide", "about 4", "0", "-3", "nan", "inf", "1e3"])
def test_a_missing_or_unusable_diameter_is_none(raw):
    assert parse_diameter_m(raw) is None


def _write_labels(path, features, crs=None):
    collection = {"type": "FeatureCollection", "features": features}
    if crs:
        collection["crs"] = {"type": "name", "properties": {"name": crs}}
    path.write_text(json.dumps(collection))
    return path


def test_polygons_multipolygons_and_points_are_read_and_lines_are_skipped(tmp_path):
    square = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
    features = [
        {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [square]}},
        {"type": "Feature", "properties": {}, "geometry": {"type": "MultiPolygon", "coordinates": [[square], [square]]}},
        {"type": "Feature", "properties": {"tags": {"diameter_crown": "6 m"}}, "geometry": {"type": "Point", "coordinates": [5, 6]}},
        {"type": "Feature", "properties": {"diameter_crown": "4"}, "geometry": {"type": "Point", "coordinates": [7, 8]}},
        {"type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [9, 9]}},
        {"type": "Feature", "properties": {}, "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}},
        {"type": "Feature", "properties": {}, "geometry": None},
    ]  # fmt: skip
    shapes, crs = read_label_shapes(_write_labels(tmp_path / "labels.geojson", features))
    assert crs == "EPSG:4326" and [shape["kind"] for shape in shapes] == ["polygon"] * 3 + ["point"] * 3
    assert [shape["diameter_m"] for shape in shapes[3:]] == [
        6.0,
        4.0,
        None,
    ]  # tags nested or flat; none given
    assert shapes[0]["bounds"] == (0.0, 0.0, 1.0, 1.0) and len(shapes[0]["rings"]) == 1


def test_the_crs_of_the_file_and_malformed_files(tmp_path):
    assert read_label_shapes(_write_labels(tmp_path / "a.geojson", [], crs="EPSG:3857"))[1] == "EPSG:3857"
    (tmp_path / "bad.geojson").write_text(json.dumps({"type": "Feature"}))
    with pytest.raises(ValueError, match="FeatureCollection"):
        read_label_shapes(tmp_path / "bad.geojson")


def test_chips_get_the_shapes_that_touch_them_in_chip_pixels(labelled_chips):
    chips, labels, names = labelled_chips
    built = build_chip_labels(chips, labels)
    assert sorted(built) == sorted(names)
    polygon, disc, ignored = built[names[0]]
    assert [polygon["kind"], disc["kind"], ignored["kind"]] == ["tree", "tree", "ignore"]
    assert polygon["rings"][0].min(axis=0).tolist() == pytest.approx([0, 0], abs=1e-3)
    assert polygon["rings"][0].max(axis=0).tolist() == pytest.approx([128, 128], abs=1e-3)
    ring = disc["rings"][0]
    assert ((ring.max(axis=0) + ring.min(axis=0)) / 2).tolist() == pytest.approx(
        [200, 60], abs=0.5
    )  # the point
    assert (ring.max(axis=0) - ring.min(axis=0)).tolist() == pytest.approx(
        [10 / GSD] * 2, rel=0.01
    )  # 10 m crown
    assert (ignored["rings"][0].max(axis=0) - ignored["rings"][0].min(axis=0))[0] == pytest.approx(
        5 / GSD, rel=0.01
    )
    assert built[names[1]][2]["kind"] == "ignore"  # every chip has the same georeferencing here


def test_a_crown_that_crosses_the_chip_edge_is_drawn_and_a_far_point_is_not(write_chip, tmp_path):
    chip = write_chip(name=CHIP_NAME)
    near_edge = tree_point(
        -10, 100, diameter_crown="10"
    )  # centre 10 px outside, radius 33 px: reaches 23 px inside
    far_away = tree_point(-90, 100, diameter_crown="10")
    labels = _write_labels(tmp_path / "edge.geojson", [near_edge, far_away])
    (found,) = build_chip_labels(chip.parent, labels)[CHIP_NAME]
    assert (
        found["rings"][0][:, 0].max() == pytest.approx(-10 + 5 / GSD, abs=0.5)
        and found["rings"][0][:, 0].min() < 0
    )


def test_a_chip_with_no_label_near_it_has_an_empty_list_and_an_ungeoreferenced_chip_is_rejected(
    write_chip, tmp_path
):
    chip = write_chip(name=CHIP_NAME)
    labels = _write_labels(
        tmp_path / "far.geojson",
        [{**tree_point(0, 0), "geometry": {"type": "Point", "coordinates": [50.0, 10.0]}}],
    )
    assert build_chip_labels(chip.parent, labels) == {CHIP_NAME: []}
    from PIL import Image

    Image.new("RGB", (256, 256)).save(tmp_path / "OAM-524288-524288-20.png")
    chip.unlink()
    with pytest.raises(ValueError, match="no CRS"):
        build_chip_labels(tmp_path, labels)


def test_load_labels_passes_a_mapping_through_and_builds_from_a_path(labelled_chips):
    chips, labels, names = labelled_chips
    mapping = {names[0]: []}
    assert load_labels(chips, mapping) is mapping
    assert sorted(load_labels(chips, labels)) == sorted(names) and sorted(
        load_labels(chips, labels / "labels.geojson")
    ) == sorted(names)


def _shape(kind, ring):
    return {"kind": kind, "rings": [np.array(ring, dtype=float)]}


def test_tree_is_drawn_over_ignore_and_holes_stay_background():
    square = [[2, 2], [18, 2], [18, 18], [2, 18], [2, 2]]
    hole = [[8, 8], [12, 8], [12, 12], [8, 12], [8, 8]]
    ignored = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
    mask = rasterize_chip_labels(
        [
            _shape("ignore", ignored),
            {"kind": "tree", "rings": [np.array(square, float), np.array(hole, float)]},
        ],
        20,
        20,
    )
    assert mask.dtype == np.uint8 and set(np.unique(mask)) == {BACKGROUND, TREE, IGNORE}
    assert (
        mask[5, 5] == TREE and mask[1, 1] == IGNORE and mask[10, 10] == BACKGROUND
    )  # tree wins; the hole stays open
    assert mask[19, 19] == BACKGROUND and not rasterize_chip_labels([], 5, 5).any()


def test_labels_are_drawn_at_the_scaled_resolution_with_sharp_edges():
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    top_left = _shape("tree", [[0, 0], [128, 0], [128, 128], [0, 128], [0, 0]])
    image, mask = scale_chip_and_labels(rgb, [top_left], PreprocessParams(), "OAM-485915-499567-20.tif")
    assert image.shape[1:] == mask.shape and image.shape[1] > 700  # zoom 20 is upscaled about 3x
    assert set(np.unique(mask)) == {BACKGROUND, TREE} and float((mask == TREE).mean()) == pytest.approx(
        0.25, abs=0.005
    )


def test_augmentation_moves_image_and_mask_together_and_keeps_all_pixels():
    rng = np.random.default_rng(3)
    mask = np.random.default_rng(5).integers(0, 3, size=(48, 48)).astype(np.uint8)  # no symmetry of its own
    image = np.stack([mask, 2 * mask, 3 * mask]).astype(np.float32)
    seen = set()
    for _ in range(40):
        out_image, out_mask = augment_chip_and_mask(image, mask, rng)
        assert out_image.flags["C_CONTIGUOUS"] and out_mask.flags["C_CONTIGUOUS"]
        assert np.array_equal(out_image[0], out_mask.astype(np.float32))  # still aligned
        assert np.array_equal(np.sort(out_mask.ravel()), np.sort(mask.ravel()))  # nothing lost or invented
        seen.add(out_mask.tobytes())
    assert len(seen) == 8  # the 4 quarter turns x mirror or not


def test_lon_lat_helper_is_consistent_with_the_chip_origin():
    assert lon_lat(0, 0) == pytest.approx([8.983, 17.676], abs=0.01)
