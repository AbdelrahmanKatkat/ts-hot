"""Chips + labels as training pairs: scaled together, cropped together, and checked at construction."""

import numpy as np
import pytest

pytest.importorskip("torch")

from ts_hot.dataset import build_chip_labels  # noqa: E402
from ts_hot.params import PreprocessParams  # noqa: E402
from ts_hot.segmentation_dataset import SegmentationDataset  # noqa: E402

PRE = PreprocessParams(
    model_gsd_m=0.1
)  # zoom 20 near the equator is 0.149 m/px: a 256 px chip scales to about 380 px


def test_a_chip_and_its_labels_come_back_at_the_same_scaled_size(labelled_chips):
    chips, labels, _ = labelled_chips
    dataset = SegmentationDataset(chips, labels, PRE)  # the GeoJSON path: the dataset builds the label shapes
    image, mask = dataset[0]
    assert (
        len(dataset) == 4 and image.shape[0] == 3 and image.shape[1:] == mask.shape and image.shape[1] > 256
    )
    assert str(image.dtype) == "torch.float32" and str(mask.dtype) == "torch.uint8"
    assert set(np.unique(mask.numpy())) == {0, 1, 255}  # polygon and tree disc, and the ignored disc


def test_a_mapping_gives_the_same_pairs_as_the_path_and_a_missing_chip_has_no_labels(labelled_chips):
    chips, labels, names = labelled_chips
    from_path = SegmentationDataset(chips, labels, PRE)[0][1]
    from_mapping = SegmentationDataset(chips, build_chip_labels(chips, labels), PRE)[0][1]
    assert np.array_equal(from_path.numpy(), from_mapping.numpy())
    assert not SegmentationDataset(chips, {}, PRE)[0][1].numpy().any()


def test_augmentation_keeps_the_size_and_valid_mask_values(labelled_chips):
    chips, labels, _ = labelled_chips
    plain, augmented = (
        SegmentationDataset(chips, labels, PRE),
        SegmentationDataset(chips, labels, PRE, augment=True, seed=1),
    )
    for index in range(4):
        (image, mask), (turned_image, turned_mask) = plain[index], augmented[index]
        assert turned_image.shape == image.shape and turned_mask.shape == mask.shape
        assert sorted(turned_mask.flatten().tolist()) == sorted(
            mask.flatten().tolist()
        )  # the same pixels, moved


def test_names_select_chips_and_a_missing_name_is_an_error(labelled_chips):
    chips, labels, names = labelled_chips
    assert [path.name for path in SegmentationDataset(chips, labels, PRE, names=names[:2]).paths] == names[:2]
    with pytest.raises(ValueError, match="not found"):
        SegmentationDataset(chips, labels, PRE, names=["OAM-1-1-20.tif"])
    with pytest.raises(ValueError, match="No chips found"):
        SegmentationDataset(labels, labels, PRE)
