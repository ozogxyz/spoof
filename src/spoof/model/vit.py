import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


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
        # better keep track of model normalization parameters, normalize by channel
        input_mean = torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None]
        input_std = torch.Tensor([0.229, 0.224, 0.225])[None, :, None, None]
        self.register_buffer("input_mean", input_mean)
        self.register_buffer("input_std", input_std)

        # Load pre-trained ViT
        self.extractor = vit_b_16(
            weights=ViT_B_16_Weights.DEFAULT,
            progress=True,
        )

        # Assign nn.Identity() to the head to be able to freeze the backbone
        self.extractor.heads = nn.Identity()

        # Replace the last layer of the transformer with custom MLP
        self.classifier = nn.Linear(dim_embedding, num_classes)

    def freeze_backbone(self):
        for param in self.extractor.parameters():
            param.requires_grad = False

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
