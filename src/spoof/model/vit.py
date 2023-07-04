import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.utils import make_grid

# NOTE: name confusion for your model and pretrained model ViT
from .lora import LoRA_ViT, ViT as pretrained_ViT
from ..utils.visualize import print_scores_on_tensor

# TODO try HTER
# TODO try HTER low res casia vga
# TODO when training casia train on low res
# TODO maybe try less blocks
# TODO try SiWM against LACC
# TODO try out_logit = out_dict["out_logit"] / 2 or 3 only in testing


LABEL_LIVE = 1


class BaseViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.debug_printed = False

    def input_check(self, x):
        if not self.debug_printed:
            log_str = f"Tensor range: [{x.min().cpu().item():.4f}, {x.max().cpu().item():.4f}]"
            class_name = self.__class__.__name__
            print(f"[{class_name}] {log_str}", flush=True)
            print(f"[{class_name}] trainable params:")
            for nam, param in self.named_parameters():
                if param.requires_grad:
                    print(nam)

            self.debug_printed = True
        return x

    @torch.no_grad()
    def plot_score_distribution(self, logger_experiment, step, dict_all):
        preds = self.get_liveness_score(dict_all).cpu().numpy().ravel()
        labels = dict_all["label"].cpu().numpy().ravel()

        preds_live = preds[labels == LABEL_LIVE]
        preds_spoof = preds[labels != LABEL_LIVE]
        phase = "train" if self.training else "val"

        if len(preds_live):
            logger_experiment.add_histogram(
                f"{phase}/preds_live", preds_live, global_step=step
            )
        if len(preds_spoof):
            logger_experiment.add_histogram(
                f"{phase}/preds_spoof", preds_spoof, global_step=step
            )

    @torch.no_grad()
    def plot_images_with_scores(
        self, logger_experiment, step, dict_all, max_img=16
    ):
        # if not self.training: return
        img = dict_all["image"].cpu().clamp(0.0, 1.0)
        preds = self.get_liveness_score(dict_all).cpu()
        labels = dict_all["label"].cpu()
        vis_imgs = print_scores_on_tensor(img, preds, labels)

        phase = "train" if self.training else "val"
        vis_grid = make_grid(vis_imgs[:max_img], 4)[[2, 1, 0], :, :]

        logger_experiment.add_image(
            f"{phase}/preds", vis_grid, global_step=step
        )

    @torch.no_grad()
    def add_tensorboard_logs(self, logger_experiment, step, dict_all):
        # helper functions for visualization
        self.verbose = True
        self.plot_score_distribution(logger_experiment, step, dict_all)
        self.plot_images_with_scores(logger_experiment, step, dict_all)


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
        # out_logit = out_dict["out_logit"] / 2 or 3, only during testing
        # check self.training
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


class LVNetVitLora(BaseViT):
    def __init__(self, num_classes: int = 1, rank=4):
        super().__init__()

        # better keep track of model normalization parameters
        input_mean = (0.5, 0.5, 0.5)  # (0.485, 0.456, 0.406)
        input_std = (0.5, 0.5, 0.5)  # (0.229, 0.224, 0.225)
        self.register_buffer(
            "input_mean", torch.Tensor(input_mean)[None, :, None, None]
        )
        self.register_buffer(
            "input_std", torch.Tensor(input_std)[None, :, None, None]
        )

        # Load pre-trained ViT
        extractor = pretrained_ViT("B_16", pretrained=True)

        # LoRa wrapper allows to replace last linear layer for classification with requested num_classes
        self.extractor = LoRA_ViT(extractor, r=rank, num_classes=num_classes)

    def get_liveness_score(self, out_dict):
        out_logit = out_dict["out_sigmoid"]
        return torch.sigmoid(out_logit)[:, 0]

    def forward(self, in_tensor):
        in_preproc = self.input_check(in_tensor)

        # normalize data
        pp_tensor = (in_preproc - self.input_mean) / self.input_std

        # feature extraction using pre-trained ViT with LoRa
        logits = self.extractor(pp_tensor)

        # generate output logits for scores
        # logits = self.flatten(logits)
        logits = torch.flatten(logits, start_dim=1)
        return {"out_sigmoid": logits}
