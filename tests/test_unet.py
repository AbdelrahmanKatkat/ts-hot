"""The tree U-Net: any input size, normalisation inside, and the two ways to load weights."""

import io

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
import torchvision  # noqa: E402

from ts_hot.export import build_unet  # noqa: E402
from ts_hot.unet import TreeUNet  # noqa: E402


def _retinanet_like_state_dict() -> dict[str, torch.Tensor]:
    """What a DeepForest checkpoint holds: a ResNet-50 body under `backbone.body.` next to FPN and head weights."""
    resnet = torchvision.models.resnet50(weights=None)
    state = {
        f"backbone.body.{key}": value
        for key, value in resnet.state_dict().items()
        if not key.startswith("fc.")
    }
    state["backbone.fpn.inner_blocks.0.0.weight"] = torch.zeros(1)
    state["head.classification_head.cls_logits.weight"] = torch.zeros(1)
    return state


def _as_stream(state: dict[str, torch.Tensor]) -> io.BytesIO:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return buffer


@pytest.mark.parametrize("height, width", [(64, 64), (97, 130), (33, 40)])
def test_output_has_the_input_size_whatever_it_is(height, width):
    logits = build_unet()(torch.rand(1, 3, height, width))
    assert logits.shape == (1, 1, height, width)


def test_normalisation_is_inside_and_not_part_of_the_saved_weights():
    model = build_unet()
    assert not any(key in model.state_dict() for key in ("mean", "std"))
    assert model.encoder.conv1.weight.shape == (64, 3, 7, 7) and not model.training


def test_the_encoder_is_initialised_from_a_retinanet_checkpoint_and_the_decoder_is_not():
    state = _retinanet_like_state_dict()
    model = build_unet(encoder_checkpoint=_as_stream(state))
    assert torch.equal(model.encoder.conv1.weight, state["backbone.body.conv1.weight"])
    assert torch.equal(model.encoder.layer4[2].conv3.weight, state["backbone.body.layer4.2.conv3.weight"])
    assert not torch.equal(
        model.head.weight, torch.zeros_like(model.head.weight)
    )  # decoder: random, not loaded


def test_a_saved_unet_loads_back_whole():
    source = build_unet()
    restored = build_unet(weights=_as_stream(source.state_dict()))
    assert all(
        torch.equal(a, b) for a, b in zip(source.state_dict().values(), restored.state_dict().values())
    )


def test_bad_checkpoints_and_both_arguments_are_rejected():
    with pytest.raises(ValueError, match="not a DeepForest state dict"):
        build_unet(encoder_checkpoint=_as_stream({"something": torch.zeros(1)}))
    with pytest.raises(ValueError, match="not both"):
        build_unet(encoder_checkpoint=_as_stream(_retinanet_like_state_dict()), weights=_as_stream({}))
    assert isinstance(build_unet(), TreeUNet)
