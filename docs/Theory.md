# Theory: tree-crown semantic segmentation on the DeepForest NEON encoder

The contract of the `ts-hot` library. Tags: **[V]** verified from source code, a checkpoint, or a run; **[P]** proposed
design decision; **[?]** unknown, needs an experiment (section 8).

## 0. Summary

| Item | Value |
|---|---|
| Task / geometry | Binary semantic segmentation (`semantic-segmentation`); the output is **polygons**, one per connected tree region, holes kept [P] |
| Model | A U-Net: the ResNet-50 body of the DeepForest NEON RetinaNet as encoder (stem and `layer1` to `layer4`, loaded from `backbone.body.*` with only `fc` missing [V]), a new decoder, 32.5 million parameters of which 9.0 million train with a frozen encoder [V] |
| Input | 3-band RGB, scaled by the zoom rule (section 3) to the 0.05 m per pixel the encoder was trained at |
| Output | A per-pixel tree probability at the input size; threshold, tracing and cleanup are outside the graph |
| Labels | Polygons and tree points in one GeoJSON, drawn into masks per chip at the model's resolution (section 2) |
| Runtime | Training on a GPU with torch; inference on a CPU with `onnxruntime`, no torch |

## 1. Task and output contract

- **Task:** every pixel of an RGB chip is tree canopy or background. Crowns that touch merge into one polygon; a crown-centre
  channel to split them is not built [?].
- **Feature schema** (`predict_session`): a `FeatureCollection` of EPSG:4326 `Polygon` features, holes kept, with the properties
  `confidence` (float, the mean tree probability over the polygon's pixels), `class_id` (int, always 1), `class_name` (`tree`) and
  `source` (the chip file name). A fAIr pack renames them to its own schema.
- **Empty prediction:** an empty `FeatureCollection`. A chip without georeferencing (PNG, JPEG) returns polygons in chip pixels.
- **Limits:** polygons are traced per chip, so a crown cut by a chip border comes back as two polygons; polygons have staircase edges
  simplified to `simplify_px`, and a simplified polygon is not checked for validity [?].

## 2. Labels and masks

A fAIr pack gets chips (`OAM-<x>-<y>-<z>.tif`, georeferenced) and one GeoJSON of labels; it does not get masks. `build_chip_labels`
turns the GeoJSON into label shapes in each chip's pixels, nothing is copied or written, and the mask is drawn on the fly at the
model's resolution (`scale_chip_and_labels`), so its edges are as sharp as the image [P].

| Feature | What it becomes |
|---|---|
| Polygon, MultiPolygon (one shape per part) | filled as **tree**, holes kept |
| Point with a usable `diameter_crown` | a disc of that diameter in metres at the chip's true ground resolution (`tile_gsd_m`), as **tree** |
| Point without a usable `diameter_crown` | a disc of 5 m as **ignore**: no reward and no penalty |
| Anything else (lines) | skipped |

- The tag is read from `properties.tags` or from the properties. Formats read: `4`, `4.5`, `4m`, `4 m`, `4 meters`, and a comma as the
  decimal (`3,5 m`); anything else, zero or a negative number is unusable [V, tested].
- Pixel priority: tree over ignore over background. A crown that crosses a chip edge is drawn from the points of the whole file, so
  both chips show it [V].
- A point is taken as longitude and latitude in degrees; a metre is 1/111,320 of a degree of latitude, within 1% [P].
- **Weak labels.** A point and a diameter are not an outline: a disc is rougher than a crown, and a tree nobody mapped is background,
  so a correct prediction there counts as an error. Scores are bounded by this noise and must be reported with it [?].
- **Check.** On the Guru dataset, masks drawn this way from `labels.geojson` match masks built independently from the dataset's own tile
  pixel maths: tree IoU 0.998 and pixel agreement 99.98% on 300 chips of each of train, validation and test [V,
  `research/analysis/check_label_masks.py`].

## 3. Preprocessing: scale from the zoom level

### 3.1 Why scale matters [V: measured on the detector built from the same encoder]

The encoder recognises crowns at a particular pixel size. The training scale is **0.05 m of ground per model pixel** (10 cm x 400 px / 800 px). Feeding a chip whose crowns are 3x smaller finds almost nothing. On the NEON sample (61 hand-annotated trees, detector F1 at IoU 0.5, conf >= 0.2), running genuine 256 px chips downsampled to a coarser resolution:

| Chip resolution | Fixed 3.125x upscale | Scale = chip GSD / 0.05 m |
|---|---|---|
| 10 cm | 0.15 | **0.61** |
| 12.5 cm | 0.42 | **0.66** |
| 15.6 cm | 0.47 | 0.49 |
| no upscale at 15.6 cm | 0.00 | n/a |

One image, in the model's training domain, with resolution emulated by area-downsampling; real coarser imagery has less detail, so these are optimistic. Whether the same scale is best for a segmentation head has not been measured [?].

### 3.2 The rule

```
chip_gsd_m = 40075016.68558 * cos(lat_of_tile_centre) / 2^z / chip_width_px     # true ground size of a chip pixel
scale      = chip_gsd_m / model_gsd_m                                              # model_gsd_m = 0.05
scale      = min(scale, max_input_px / longest chip side)                         # max_input_px = 1024
input size = round(chip size * scale)                                              # one bilinear resize, in float
```

`z`, and `y` (which gives the latitude) come from the chip name `OAM-<x>-<y>-<z>.<ext>` (`ts_hot.tiles`). Nothing is read from the file, so PNG/JPEG chips work the same as GeoTIFFs. Examples for a 256 px chip: zoom 20 at the equator is 14.9 cm/px -> scale 2.99; the same zoom at 48 deg latitude is 10 cm/px -> scale 2.0 [V: formula, unit-tested].

`PreprocessParams.model_gsd_m` (default 0.05, range [0.005, 1]) sets the scale and `max_input_px` (default 1024, range [64, 8192]) caps it: zoom 20 is untouched (about 760 px), zoom 19 goes from 5.9 to 4.0 at Freetown, zoom 18 from 10.6 to 4.0. **Training and serving must use the same value** [P]. Scales above 16 (a 256 px chip coarser than 80 cm/px) raise `ValueError`.

There is **no padding and no crop** at serving: the network takes any height and width, and the decoder resizes to the size of the skip features it joins, so the output has the input's size [V: ONNX matches torch at 97x130, 33x200, 64x64 and 160x168].

### 3.3 Is the zoom always known?

- OpenAerialMap itself has no zoom: a TMS endpoint `.../{z}/{x}/{y}` serves any zoom, and zoom exists only in the tile request.
- fAIr chips carry it: they are named `OAM-{x}-{y}-{z}.tif`.
- Zoom gives the *nominal* resolution. If the source imagery is coarser than the zoom, tiles are upsampled and look blurry: the scale is then right but the detail is not.

### 3.4 Normalisation on OAM imagery

24 recent OAM images, against ImageNet and the NEON sample: OAM is slightly brighter and lower-contrast than ImageNet, but it varies between images by more than it differs from ImageNet, and the NEON imagery the encoder was trained on is further from ImageNet than OAM is. Decision [P]: **keep the ImageNet constants fixed**, inside the model (`TreeUNet` normalises its own input, the constants are not in the weights file), so training and serving cannot disagree.

## 4. Model and ONNX contract

- **Encoder:** ResNet-50 (stem, `layer1` to `layer4`, strides 2, 4, 8, 16, 32). `build_unet(encoder_checkpoint=...)` loads the `backbone.body.*` weights of a DeepForest checkpoint (`NEON.pt`, `NEON_OAM.pt`) strictly; the FPN and detection heads are not used. The stem and `layer1` to `layer3` of `NEON_OAM` are identical to `NEON`; only `layer4` differs, by about 2% [V].
- **Decoder:** five up-blocks (256, 128, 64, 32, 16 channels), each upsampling to the size of the skip features, joining them, and two 3x3 convolutions with BatchNorm; the last one upsamples to the input size and a 1x1 convolution gives the logit [V].
- **ONNX (opset 17, legacy exporter):** input `image` `(3, H, W)` float32 in [0, 1] RGB with dynamic `H`, `W`; output `probability` `(H, W)` float32 in [0, 1]. Normalisation, the network and the sigmoid are inside; the graph is 130 MB [V].
- **Level of the graph:** the network and the fixed steps are inside; the tunable steps (threshold, tracing, minimum area, simplification) are outside, as for the detector [P].
- **Parity:** the exported graph and the torch model agree to 1e-4 on random weights at four sizes, and score the same IoU on toy chips to within 0.01 [V, synthetic]; parity on a trained model and on real chips is [?].

## 5. Decoding and map-space postprocessing

Model space (`decode.py`, no CRS or affine): threshold the probability; trace the pixel edges of the foreground (4-connectivity, so pixels that only touch at a corner are separate; a region with a hole is one polygon with an interior ring); drop polygons under `min_area_px`; simplify each ring with Douglas-Peucker at `simplify_px` (a ring that collapses to fewer than three vertices is dropped); attach the mean probability over the polygon's pixels as `confidence` [V, tested].

Map space (`georef.py`, `postprocess.py`, no network layout): divide the vertices by the preprocessing scale to get chip pixels; apply the chip's affine transform; convert to EPSG:4326 with `pyproj`. Thresholds are in model-input pixels, so they do not depend on the zoom or the CRS; `min_area_px = 400` is 1 m2 at 0.05 m [P]. No shapely is used; whether to add it for merging polygons across chips is decided later [?].

## 6. Parameters

Keys are bare; fAIr strips the `training.` and `inference.` prefixes. `parse_*` tolerates `None` and missing keys.

| Training key | Type | Default | Range | Purpose |
|---|---|---|---|---|
| `epochs` | int | 5 | >= 1 | passes over the training chips |
| `batch_size` | int | 2 | >= 1 | chips per batch (the chips of a dataset must scale to one size, that is, one zoom) |
| `learning_rate` | float | 1e-3 | > 0 | AdamW |
| `weight_decay` | float | 1e-4 | >= 0 | AdamW |
| `scheduler`, `warmup_steps` | str, int | `cosine`, 0 | `cosine` or `none`; >= 0 | per-batch schedule |
| `max_grad_norm` | float | 1.0 | >= 0 | gradient clipping, 0 disables |
| `freeze_encoder`, `unfreeze_from`, `encoder_lr_factor` | bool, str, float | true, `none`, 1.0 | `layer1` to `layer4`; (0, 1] | the encoder stays frozen unless asked; BatchNorm of the encoder never trains |
| `epoch_fraction` | float | 1.0 | (0, 1] | a fresh random share of the chips each epoch |
| `early_stop_patience` | int | 5 | >= 1 | epochs without a better validation loss |
| `dice_weight` | float | 1.0 | >= 0 | weight of the Dice loss next to binary cross-entropy |
| `val_ratio`, `split_seed`, `block_size` | float, int, int | 0.2, 42, 4 | | the spatial split (section 7) |
| `model_gsd_m`, `max_input_px` | float, int | 0.05, 1024 | see 3.2 | the scale; must match serving |

| Inference key | Type | Default | Range | Stage |
|---|---|---|---|---|
| `threshold` (alias `confidence_threshold`) | float | 0.5 | [0, 1] | decoding: a pixel is tree from this probability |
| `min_area_px` | int | 400 | >= 0 | decoding: smaller polygons are dropped |
| `simplify_px` | float | 1.0 | >= 0 | decoding: Douglas-Peucker tolerance, 0 keeps every pixel edge |
| `model_gsd_m`, `max_input_px` | | | | preprocessing |

Device: CUDA only when visible, otherwise CPU (`device = auto`); serving is CPU with `onnxruntime`. The loss is binary cross-entropy plus Dice over the pixels that are not ignored; a fine-tuned model is not calibrated like the base, so the `threshold` is chosen by a sweep on validation chips (`sweep_threshold`), never reused [P].

## 7. Spatial split for training / validation

Neighbouring chips share canopy and trees cut at chip borders, so a random per-chip split leaks near-duplicates into validation and inflates metrics. `ts_hot.dataset.spatial_split(chip_names, val_ratio, seed, block_size=4)`:

1. parse OAM tile names `OAM-<x>-<y>-<z>.ext`;
2. group chips into blocks `(x // block_size, y // block_size)`;
3. shuffle blocks with `numpy.default_rng(seed)`; fill validation with **whole blocks** until `val_ratio` is reached; the rest is train;
4. a name not matching the pattern raises `ValueError`.

Deterministic for a given seed, disjoint, full cover [V, tested]. A re-fine-tune on a new dataset may put earlier validation chips into training, so metrics are only comparable across generations if the held-out set is fixed [P]. The Guru dataset ships its own train, validation and test regions; those are used as they are.

## 8. Verification and open questions

**Verified here:** the encoder loads from `NEON_OAM.pt`; the ONNX export and its dynamic sizes; the label-to-mask rules against an independent construction; a training run end to end on toy chips (train, export, score through torch and ONNX); a pilot on the real dataset (1% of an epoch: 0.28 s per step at batch 4, 1.4 GB of GPU memory, loss falling from about 1.6 to 1.2).

**Not measured yet [?]:**

- the quality of a trained model: pixel IoU and F1 on the validation and test splits, and the best threshold;
- a baseline: the detector's boxes drawn as masks, to show what the segmentation head adds;
- forgetting: there are no source-domain masks for NEON, so only the loss on the target can be reported;
- the best scale for segmentation, the effect of `layer4` fine-tuning against a frozen encoder, and NEON against NEON_OAM as the encoder;
- other cities, sensors and zooms; parity of the exported graph on a trained model and on real chips.

**Sources:** the DeepForest paper and package (the encoder, its scale), torchvision (ResNet-50), the fAIr contributing guide (chips, labels and tasks), and the Guru dataset `hotosm/fAIr-guru-td-trees` (its measured composition is in `research/datasets.md`, local).
