import torch

from src.spoof.model.vit import LVNetVitLora


def test_lora():
    model = LVNetVitLora()
    model.eval()

    x = torch.rand(1, 3, 224, 224)
    out = model(x)

    pred = model.get_liveness_score(out)

    pred = pred.detach().cpu().numpy()
    print(pred)
    assert pred.max() <= 1.0
    assert pred.min() >= 0.0

    assert False
