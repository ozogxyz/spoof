import torch

from spoof.model.vit import ViT


def test_model():
    img = torch.rand(1, 3, 224, 224)
    model = ViT()
    out = model(img)

    liveness_score = model.get_liveness_score(out)

    assert liveness_score.shape == (1, 1)
    assert liveness_score.max() <= 1.0
    assert liveness_score.min() >= 0.0
