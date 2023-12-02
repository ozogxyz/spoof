import pytest
import torch


@pytest.fixture
def model_transformer():
    model = torch.load("logs/ep004_loss0.00_acc1.000_eer0.000ckpt", map_location="cpu")[
        "state_dict"
    ]
    # Pop the mean and std buffers
    model.pop("model.input_mean")
    model.pop("model.input_std")

    return model


@pytest.fixture
def vit_transformer():
    return torch.load(
        "/Users/motorbreath/mipt/spoof/.cache/torch/hub/checkpoints/vit_b_16-c867db91.pth",
        map_location="cpu",
    )


def test_transformer_weights(model_transformer, vit_transformer):
    model_ln = model_transformer["model.extractor.encoder.ln.weight"]
    vit_ln = vit_transformer["encoder.ln.weight"]
    assert torch.allclose(model_ln, vit_ln, atol=1e-5)


def test_classifier_weights(model_transformer, vit_transformer):
    model_ln = model_transformer["model.classifier.weight"]
    vit_ln = vit_transformer["heads.head.weight"]
    assert not torch.allclose(model_ln, vit_ln, atol=1e-5)
