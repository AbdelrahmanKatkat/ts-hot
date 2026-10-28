"""Chip-pixel polygons -> GeoJSON FeatureCollection."""

from typing import Any

from .georef import ring_to_epsg4326
from .preprocess import PreprocessMeta

CLASS_NAMES = {1: "tree"}


def polygons_to_feature_collection(
    polygons: list[dict[str, Any]],
    meta: PreprocessMeta,
    *,
    source_name: str | None = None,
) -> dict[str, Any]:
    """`polygons` are in source-chip pixels; output is EPSG:4326 polygons with their holes."""
    features: list[dict[str, Any]] = []
    for polygon in polygons:
        rings = [ring_to_epsg4326(ring, meta.transform, meta.crs) for ring in polygon["rings"]]
        props: dict[str, Any] = {
            "confidence": round(float(polygon["confidence"]), 4),
            "class_id": int(polygon["class_id"]),
            "class_name": CLASS_NAMES.get(int(polygon["class_id"]), "tree"),
        }
        if source_name:
            props["source"] = source_name
        features.append(
            {"type": "Feature", "properties": props, "geometry": {"type": "Polygon", "coordinates": rings}}
        )
    return {"type": "FeatureCollection", "features": features}
