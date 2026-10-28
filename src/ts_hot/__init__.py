"""`ts-hot`: HOT fAIr tree-crown semantic segmentation helpers (U-Net on the DeepForest NEON ResNet-50 encoder)."""

from .dataset import spatial_split
from .decode import decode_polygons
from .params import (
    PostprocessParams,
    PreprocessParams,
    SplitParams,
    TrainParams,
    parse_postprocess_params,
    parse_preprocess_params,
    parse_split_params,
    parse_train_params,
)
from .postprocess import polygons_to_feature_collection
from .preprocess import PreprocessMeta, preprocess_chip, preprocess_rgb
from .serve import predict_chip, predict_session
from .tiles import chip_scale, tile_gsd_m

__all__ = [
    # parameters
    "PreprocessParams",
    "PostprocessParams",
    "SplitParams",
    "TrainParams",
    "parse_preprocess_params",
    "parse_postprocess_params",
    "parse_split_params",
    "parse_train_params",
    # tiles and dataset
    "tile_gsd_m",
    "chip_scale",
    "spatial_split",
    # preprocess -> decode -> postprocess
    "PreprocessMeta",
    "preprocess_rgb",
    "preprocess_chip",
    "decode_polygons",
    "polygons_to_feature_collection",
    # inference
    "predict_chip",
    "predict_session",
]
# Optional (train extra, import explicitly): ts_hot.export, ts_hot.unet, ts_hot.segmentation_dataset, ts_hot.train, ts_hot.evaluate
