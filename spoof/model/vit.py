import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from timm import create_model


class ViT(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        dim_embedding: int = 768,
    ):
        """
        Args:
            `model`: The transformer model.
            `n_classes`: Number of classes in the classification task.
        """
        super().__init__()
        # better keep track of model normalization parameters
        half_ones = torch.Tensor([0.5])[None, None, None, :]
        self.register_buffer("input_mean", half_ones)
        self.register_buffer("input_std", half_ones)

        # Load pre-trained ViT
        self.extractor = create_model("vit_base_patch16_224", pretrained=True)
        self.extractor.head = nn.Identity()

        # Freeze the transformer
        for param in self.extractor.parameters():
            param.requires_grad = False

        # Replace the last layer of the transformer with custom MLP
        self.classifier = nn.Linear(dim_embedding, num_classes)

    def get_liveness_score(self, out_dict):
        out_logit = out_dict["out_logit"]
        return torch.sigmoid(out_logit)

    def forward(self, in_tensor):
        # normalize data
        pp_tensor = (in_tensor - self.input_mean) / self.input_std

        # feature extraction using pre-trained ViT
        features = self.extractor(pp_tensor)

        # generate output logits for scores
        logits = self.classifier(features)
        logits = torch.flatten(logits, start_dim=1)
        return {
            "out_logit": logits,
        }
