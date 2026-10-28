"""Chips + labels as a torch dataset for U-Net fine-tuning (needs the `train` extra; import explicitly)."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .dataset import (
    augment_chip_and_mask,
    list_chip_paths,
    load_labels,
    scale_chip_and_labels,
)
from .params import PreprocessParams
from .preprocess import read_chip


class SegmentationDataset(torch.utils.data.Dataset):
    """Chips + their label shapes (`dataset.build_chip_labels`) as `(image, mask)` pairs.

    Each chip is scaled exactly as at serving time and its labels are drawn at that resolution
    (`scale_chip_and_labels`); with `augment`, the pair is also turned by a random quarter turn and mirrored
    (`augment_chip_and_mask`). The whole scaled chip is returned, so the chips of one batch must scale to one size.
    `labels` is the mapping from `build_chip_labels`, or the GeoJSON path to build it from; a chip missing from the
    mapping counts as having no labels.
    """

    def __init__(
        self,
        images_dir: str | Path,
        labels: str | Path | Mapping[str, list[dict[str, Any]]],
        pre: PreprocessParams,
        *,
        names: Sequence[str] | None = None,
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        self.pre = pre
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.paths = list_chip_paths(images_dir)
        if names is not None:
            wanted = set(names)
            self.paths = [path for path in self.paths if path.name in wanted]
            missing = wanted - {path.name for path in self.paths}
            if missing:
                raise ValueError(
                    f"{len(missing)} named chips not found in {images_dir}, e.g. {sorted(missing)[:3]}"
                )
        if not self.paths:
            raise ValueError(f"No chips found in {images_dir}")
        self.labels = load_labels(images_dir, labels)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[idx]
        rgb, _, _ = read_chip(path)
        image, mask = scale_chip_and_labels(rgb, self.labels.get(path.name, []), self.pre, path.name)
        if self.augment:
            image, mask = augment_chip_and_mask(image, mask, self.rng)
        return torch.from_numpy(image), torch.from_numpy(mask)
