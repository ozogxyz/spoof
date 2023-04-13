import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer


class ViT(nn.Module):
    def __init__(
        self,
        model: VisionTransformer,
        n_classes: int,
    ):
        """
        Args:
            `model`: The transformer model.
            `n_classes`: Number of classes in the classification task.
        """
        super().__init__()
        self.model = model
        self.classifier = nn.Linear(model.num_features, n_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x
