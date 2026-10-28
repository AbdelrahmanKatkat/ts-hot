"""`predict_session`: directory of OAM chips -> scale -> ONNX -> polygons -> GeoJSON."""

import logging
import time
from pathlib import Path
from typing import Any

from .decode import decode_polygons
from .georef import polygons_to_chip_pixels
from .params import parse_postprocess_params, parse_preprocess_params
from .postprocess import polygons_to_feature_collection
from .preprocess import PreprocessMeta, preprocess_chip

log = logging.getLogger(__name__)


def list_images(input_dir: str | Path) -> list[Path]:
    """List chip files in a directory (png/tif/jpg)."""
    root = Path(input_dir)
    patterns = ("*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg")
    files = sorted(path for pattern in patterns for path in root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No input images found in {root}")
    return files


def segment_chip(
    session: Any, image_path: str | Path, params: dict[str, Any] | None
) -> tuple[list[dict[str, Any]], PreprocessMeta]:
    """One chip through the ONNX session: polygons in source-chip pixels, plus the metadata to georeference them."""
    pre_cfg = parse_preprocess_params(params)
    post_cfg = parse_postprocess_params(params)

    image, meta = preprocess_chip(image_path, pre_cfg)
    log.debug(
        "%s: scale %.3f -> input %dx%d",
        Path(image_path).name,
        meta.scale_x,
        meta.resized_width,
        meta.resized_height,
    )
    (probability,) = session.run(None, {session.get_inputs()[0].name: image})[:1]
    polygons = decode_polygons(probability, post_cfg)
    return polygons_to_chip_pixels(polygons, meta), meta


def predict_chip(session: Any, image_path: str | Path, params: dict[str, Any] | None) -> dict[str, Any]:
    """Predict one chip (a single model pass) and return its FeatureCollection."""
    polygons, meta = segment_chip(session, image_path, params)
    return polygons_to_feature_collection(polygons, meta, source_name=Path(image_path).name)


def predict_session(session: Any, input_images: str | Path, params: dict[str, Any] | None) -> dict[str, Any]:
    """Run ONNX prediction for all chips in a directory."""
    features: list[dict[str, Any]] = []
    started = time.time()
    paths = list_images(input_images)
    for img_path in paths:
        features.extend(predict_chip(session, img_path, params)["features"])
    log.info(
        "predict_session: %d chips, %d polygons in %.1fs", len(paths), len(features), time.time() - started
    )
    return {"type": "FeatureCollection", "features": features}
