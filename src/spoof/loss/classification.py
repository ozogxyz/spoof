import torch
from torch.nn import functional as F

from .base import LossModule


class ClassifierBinary(LossModule):
    """
    this is an example class for binary classification loss
    """

    def __init__(
        self,
        name="bce",
        with_logits=True,
        tag_pred="out_logit",
        tag_gt="label",
    ):
        super().__init__(name, tag_pred, tag_gt)
        self.with_logits = with_logits

    def forward(self, **allvars):
        pred = allvars.get(self.tag_pred)
        label = allvars.get(self.tag_gt)

        b = label.size(0)
        label_copy = (label.clone().float().view(b, -1) > 0.5).float()
        out_sigmoid = pred.view(b, -1)
        if self.with_logits:
            loss = F.binary_cross_entropy_with_logits(out_sigmoid, label_copy)
        else:
            loss = F.binary_cross_entropy(out_sigmoid, label_copy)
        return loss


class ClassifierMulti(LossModule):
    """
    this is an example class for multiclass classification loss
    """

    def forward(self, **allvars):
        logits = allvars.get(self.tag_pred)
        labels = allvars.get(self.tag_gt)

        pred_lsm = F.log_softmax(logits, dim=1)

        pred_lsm = torch.flatten(pred_lsm, start_dim=1)
        labels = torch.flatten(labels).long()

        loss_value = F.nll_loss(pred_lsm, labels)
        return loss_value
