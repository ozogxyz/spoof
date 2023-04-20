from torch import nn


class LossModule(nn.Module):
    def __init__(self, name, tag_pred, tag_gt):
        super().__init__()
        self.name = name
        self.tag_pred = tag_pred
        self.tag_gt = tag_gt
