import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from timm import create_model


class ViT(nn.Module):
    def __init__(
        self,
        n_classes: int = 2,
    ):
        """
        Args:
            `model`: The transformer model.
            `n_classes`: Number of classes in the classification task.
        """
        super().__init__()
        self.model = create_model("vit_base_patch16_224", pretrained=True)
        self.model.head = nn.Identity()
        self.classifier = nn.Linear(768, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        x = self.softmax(x)

        return x
