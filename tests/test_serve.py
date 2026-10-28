"""`predict_session` end to end with a fake ONNX session: chips in, georeferenced polygons out."""

import numpy as np
import pytest
from PIL import Image

from conftest import CHIP_NAME
from ts_hot.params import PostprocessParams
from ts_hot.serve import list_images, predict_chip, predict_session, segment_chip


class FakeSession:
    """Stands in for onnxruntime: returns no tree, or one planted block, as the `probability` output."""

    class _In:
        name = "image"

    def __init__(self, plant: bool = False):
        self.plant = plant
        self.calls: list[tuple[int, ...]] = []

    def get_inputs(self):
        return [self._In()]

    def run(self, _outputs, feeds):
        image = feeds["image"]
        assert image.ndim == 3 and image.dtype == np.float32 and 0.0 <= image.min() <= image.max() <= 1.0
        self.calls.append(image.shape)
        probability = np.zeros(image.shape[1:], dtype=np.float32)
        if self.plant:
            probability[300:600, 150:450] = 0.9  # model-input pixels: rows 300-600, columns 150-450
        return [probability]


def test_list_images_is_sorted_and_an_empty_folder_is_an_error(tmp_path):
    (tmp_path / "b.png").write_bytes(b"")
    (tmp_path / "a.tif").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    assert [path.name for path in list_images(tmp_path)] == ["a.tif", "b.png"]
    with pytest.raises(FileNotFoundError, match="No input images"):
        list_images(tmp_path / "missing")


def test_a_polygon_comes_back_in_chip_pixels(write_chip):
    session = FakeSession(plant=True)
    polygons, meta = segment_chip(session, write_chip(name=CHIP_NAME), None)
    assert (
        session.calls == [(3, meta.resized_height, meta.resized_width)] and meta.resized_width > 700
    )  # scaled ~3x
    (polygon,) = polygons
    exterior = polygon["rings"][0]
    scale = meta.scale_x
    assert exterior.min(axis=0) == pytest.approx([150 / scale, 300 / scale], abs=1.0)  # back in chip pixels
    assert exterior.max(axis=0) == pytest.approx([450 / scale, 600 / scale], abs=1.0)
    assert polygon["confidence"] == pytest.approx(0.9, abs=1e-6) and polygon["class_id"] == 1


def test_predict_chip_gives_georeferenced_features(write_chip):
    collection = predict_chip(FakeSession(plant=True), write_chip(name=CHIP_NAME), None)
    (feature,) = collection["features"]
    assert feature["properties"] == {
        "confidence": 0.9,
        "class_id": 1,
        "class_name": "tree",
        "source": CHIP_NAME,
    }
    ring = np.array(feature["geometry"]["coordinates"][0])
    assert (np.abs(ring[:, 0] - 8.98) < 0.01).all() and (np.abs(ring[:, 1] - 17.68) < 0.05).all()  # lon, lat


def test_predict_session_collects_every_chip_and_empty_or_missing_params_are_safe(write_chip, tmp_path):
    write_chip(name="OAM-524288-524288-20.tif")
    write_chip(name="OAM-524289-524288-20.tif")
    session = FakeSession(plant=True)
    collection = predict_session(session, tmp_path, {})
    assert len(session.calls) == 2 and len(collection["features"]) == 2
    assert {f["properties"]["source"] for f in collection["features"]} == {
        "OAM-524288-524288-20.tif",
        "OAM-524289-524288-20.tif",
    }
    assert predict_session(FakeSession(plant=False), tmp_path, None) == {
        "type": "FeatureCollection",
        "features": [],
    }


def test_params_change_the_decoding_not_the_model_call(write_chip, tmp_path):
    write_chip(name=CHIP_NAME)
    assert len(predict_session(FakeSession(plant=True), tmp_path, {"threshold": 0.95})["features"]) == 0
    assert (
        len(predict_session(FakeSession(plant=True), tmp_path, {"confidence_threshold": 0.5})["features"])
        == 1
    )
    big = {"min_area_px": int(PostprocessParams().min_area_px) * 1000}
    assert len(predict_session(FakeSession(plant=True), tmp_path, big)["features"]) == 0


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
def test_a_png_chip_keeps_chip_pixel_coordinates(tmp_path):
    Image.new("RGB", (256, 256)).save(tmp_path / CHIP_NAME.replace(".tif", ".png"))
    (feature,) = predict_session(FakeSession(plant=True), tmp_path, None)["features"]
    ring = np.array(feature["geometry"]["coordinates"][0])
    lower, upper = (
        ring.min(axis=0),
        ring.max(axis=0),
    )  # the scale is 2.98: columns 150-450 -> 50.3-151, rows 300-600 -> 100.7-201.3
    assert lower == pytest.approx([50.3, 100.7], abs=1.0) and upper == pytest.approx([151.0, 201.3], abs=1.0)
