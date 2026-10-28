# `ts-hot`

Python helper library for the HOT **tree-crown semantic segmentation** model on fAIr: a U-Net decoder on the ResNet-50 encoder of the
[`weecology/deepforest-tree`](https://huggingface.co/weecology/deepforest-tree) detector (DeepForest NEON, or its Freetown fine-tune
`NEON_OAM`). It is the sibling of `td-hot`, the detector library, and follows the same split: this library holds the model logic, and a
fAIr model pack (`fAIr-models/models/<pack>/`) stays a thin STAC/ZenML/Docker adapter.

**Python:** >= 3.12. Design decisions and the contract are in [`docs/Theory.md`](docs/Theory.md).

## Inference path

```text
chip dir (OAM-<x>-<y>-<z>.tif/.png/.jpg, RGB)
  -> preprocess_chip()           one resize; scale = chip ground resolution (from the zoom and tile row) / 0.05 m
  -> onnxruntime session.run()   ONNX: normalise, U-Net, sigmoid -> probability (H, W), any size
  -> decode_polygons()           threshold, trace the pixel edges, drop small polygons, simplify
  -> polygons_to_chip_pixels()   back to chip pixels
  -> polygons_to_feature_collection()   EPSG:4326 polygons (holes kept)
```

Entry point: `ts_hot.predict_session(session, input_images, params)` returns a GeoJSON `FeatureCollection`
(`confidence`, `class_id`, `class_name`, `source` per feature). All `params` are optional; see `docs/Theory.md` section 6.

**Chip names must be `OAM-<x>-<y>-<z>.<ext>`.** The zoom and tile row give the true ground resolution, so the scale is right even for
PNG/JPEG chips with no georeferencing (their polygons stay in chip pixels). A zoom that would need an upscale above 16x is rejected, and the
scaled chip is capped at `max_input_px` (default 1024) on its longest side. Polygons are traced per chip, so a crown cut by a chip border
comes back as two polygons.

## Install, extras and who installs what

| Extra | Contents | Who installs it |
|---|---|---|
| *(none)* | numpy, Pillow, pyproj, rasterio. **No torch.** | the CPU inference image (plus `onnxruntime`, which the fAIr serving base provides) |
| `train` | torch, torchvision, onnx: fine-tuning, evaluation and ONNX export | the fAIr **GPU** training image |
| `infer` | onnxruntime | running an exported ONNX locally (`scripts/predict.py`) |
| `test` | onnxruntime, pytest | CI |
| `dataprep` | datasets (Hugging Face) | **this repository only** (`research/`); fAIr-models must never install it |

Training always runs on a GPU, but the CUDA build of torch is chosen by the image (for example
`--extra-index-url https://download.pytorch.org/whl/cu126`), not by this package's metadata.

```bash
uv sync                                # inference core
uv sync --extra train --extra test     # development
```

A fAIr model pack installs the library from a pinned release tag, for example
`uv pip install "ts-hot[train] @ git+https://github.com/<owner>/ts-hot@<tag>"` in the training stage and
`"ts-hot @ git+https://github.com/<owner>/ts-hot@<tag>"` in the inference stage. The repository has to be reachable without credentials,
and the tag must exist.

ONNX I/O: input `image (3, H, W)` float32 in [0, 1] with dynamic `H`, `W`; output `probability (H, W)` float32 in [0, 1]. Normalisation, the
network and the sigmoid are inside the graph; the threshold and the polygon steps are in `decode_polygons`.

## Layout

| Module | Role |
|---|---|
| `params.py` | defaults + `parse_*` (tolerate `None`, aliases `confidence_threshold`, `lr`) |
| `preprocess.py` | the one scale step, `PreprocessMeta` |
| `tiles.py` | OAM chip name -> ground resolution -> scale (zoom and latitude, no CRS) |
| `decode.py` | probability -> polygons in model pixels: threshold, trace, minimum area, simplify (no CRS) |
| `georef.py`, `postprocess.py` | model -> chip pixels -> EPSG:4326, GeoJSON |
| `serve.py` | `predict_session`, `predict_chip`, `segment_chip` |
| `dataset.py` | `spatial_split`; `build_chip_labels` (a labels GeoJSON -> label shapes in each chip's pixels); the per-chip steps of training (draw the labels at the model's scale, crop, augment) |
| `unet.py` | `TreeUNet`: ResNet-50 encoder, U-Net decoder, normalisation inside (`train`) |
| `export.py` | `build_unet` (owns the checkpoints), `probability_network`, the ONNX export (`train`) |
| `segmentation_dataset.py`, `train.py` | chips + labels as a torch dataset; fine-tuning (`train`) |
| `evaluate.py` | pixel precision / recall / F1 / IoU, threshold sweep (torch only for `score_dataset`) |

`unet`, `export`, `segmentation_dataset` and `train` need the `train` extra and are imported explicitly (`from ts_hot.train import ...`);
`import ts_hot`, `serve`, `evaluate`, `decode` and `export` never load torch until a function that needs it is called.

## Scripts

`scripts/` holds thin command-line entry points; each one only parses arguments and calls `ts_hot`.

| Script | Calls | Extra |
|---|---|---|
| `scripts/finetune.py` | `ts_hot.train` + `export`: fine-tunes, writes `<name>.pt` and `<name>.onnx`, `train.log`, `history.json`, one checkpoint per epoch | `train` |
| `scripts/evaluate.py` | `ts_hot.evaluate`: pixel P/R/F1/IoU and a threshold sweep of torch weights or of the deployed ONNX | `train` (`--weights`) or `infer` (`--onnx`) |
| `scripts/predict.py` | `ts_hot.predict_session`: an ONNX over a folder of chips to GeoJSON, on CPU | `infer` |
| `scripts/export_onnx.py` | `ts_hot.export`: a `<name>.pt` to an ONNX file | `train` |

A dataset folder for `finetune.py` holds `chips/` (georeferenced GeoTIFFs named `OAM-<x>-<y>-<z>.tif`) and `labels/` (one GeoJSON of polygons
and tree points): the layout a fAIr pack receives. `research/analysis/make_guru_labels.py` builds it for the Guru dataset from the labels the
dataset already carries (local, not committed).

## Labels

Masks are not files: `build_chip_labels` turns the GeoJSON into label shapes in each chip's pixels, and the training and scoring code draws
them at the model's resolution. A polygon is filled as tree. A point is a disc of its `diameter_crown` metres (read from `properties.tags`
or the properties, as `4`, `4 m`, `4 meters` or `3,5 m`), also as tree. A point **without** a usable diameter is a disc of 5 m that is *ignored*:
the loss and the scores skip it, and its tile stays in the dataset. Tree wins over ignore, lines are skipped, and a crown that crosses a chip
edge is drawn from the points of the whole file. Chips must be georeferenced and named `OAM-<x>-<y>-<z>`. The rules and their limits are in
`docs/Theory.md` section 2.

## From a pipeline step (what a fAIr-models pack calls)

```python
from ts_hot import parse_split_params, parse_preprocess_params, spatial_split
from ts_hot.dataset import build_chip_labels
from ts_hot.params import parse_train_params
from ts_hot.train import train                    # `train` extra, imported inside the step
from ts_hot.export import export_onnx_bytes
from ts_hot.evaluate import score_session         # no torch: scores the exported ONNX

split = parse_split_params(hyperparameters)       # val_ratio, split_seed, block_size
labels = build_chip_labels(chips_dir, labels_geojson)  # chip name -> label shapes in chip pixels
train_names, val_names = spatial_split(sorted(labels), split.val_ratio, split.split_seed, block_size=split.block_size)
result = train(
    train_images_dir=chips_dir, train_labels=labels, train_names=train_names,
    val_images_dir=chips_dir, val_labels=labels, val_names=val_names,
    cfg=parse_train_params(hyperparameters), pre=parse_preprocess_params(hyperparameters),
    checkpoint=base_weights, on_epoch_end=lambda epoch, train_loss, val_loss: ...)  # -> MLflow
onnx_bytes = export_onnx_bytes(result.model)
metrics = score_session(onnx_session, images_dir=chips_dir, labels=labels, names=val_names)[0.5]
```

`checkpoint` is a path or an open binary stream (for example `io.BytesIO(weights_bytes)`) of a U-Net state dict; to start a new U-Net from a
DeepForest checkpoint give `encoder_checkpoint` instead. At serving time the pack calls `ts_hot.predict_session(session, input_images, params)`;
`ts_hot.serve.segment_chip` returns the polygons in chip pixels when a step needs them instead of GeoJSON.

Each step rebuilds the labels from the chips and the GeoJSON it was given (about 5 ms per chip), so nothing is written or shared between
isolated steps.

## Tests

```bash
uv run ruff check src scripts tests
uv run pytest -q                      # offline: toy chips, random weights, a fake ONNX session
```
