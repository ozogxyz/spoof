import torch
from torch import nn


class DummyModel(nn.Module):
    """
    this is a dummy example of model class
    """

    def __init__(self, num_classes=1, dim_embedding=16):
        super().__init__()
        # better keep track of model normalization parameters
        half_ones = torch.Tensor([0.5])[None, None, None, :]
        self.register_buffer("input_mean", half_ones)
        self.register_buffer("input_std", half_ones)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, dim_embedding, kernel_size=1),
        )
        self.classifier = nn.Conv2d(dim_embedding, num_classes, kernel_size=1)

    def get_liveness_score(self, out_dict):
        out_logit = out_dict["out_logit"]
        return torch.sigmoid(out_logit)

    def forward(self, in_tensor):
        # normalize data
        pp_tensor = (in_tensor - self.input_mean) / self.input_std

        features = self.extractor(pp_tensor)
        # squeeze H and W
        embs = self.gap(features)

        # generate output logits for scores
        logits = self.classifier(embs)
        logits = torch.flatten(logits, start_dim=1)
        return {
            "out_logit": logits,
        }
