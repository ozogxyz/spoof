from torch import nn


class MultiLoss(nn.Module):
    """Combines several loss functions for convenience.
    Stolen from one of the repos on keypoint detection.
    *args: [loss weight (float), loss object, ... ]

    Example:
        loss = MultiLoss( 1, MyFirstLoss(), 0.5, MySecondLoss() )
    """

    def __init__(self, weighted_loss_list):
        super().__init__()
        assert (
            len(weighted_loss_list) % 2 == 0
        ), "args must be a list of (float, loss)"
        self.weights = []
        self.losses = nn.ModuleList()
        for i in range(len(weighted_loss_list) // 2):
            weight = float(weighted_loss_list[2 * i + 0])
            loss = weighted_loss_list[2 * i + 1]
            assert isinstance(loss, nn.Module), "%s is not a loss!" % loss
            self.weights.append(weight)
            self.losses.append(loss)

    def forward(self, select=None, **variables):
        assert not select or all(1 <= n <= len(self.losses) for n in select)
        details = dict()
        cum_loss = 0
        for num, (weight, loss_func) in enumerate(
            zip(self.weights, self.losses), 1
        ):
            if select is not None and num not in select:
                continue
            loss_value = loss_func(**{k: v for k, v in variables.items()})
            if isinstance(loss_value, tuple):
                assert len(loss_value) == 2 and isinstance(loss_value[1], dict)
            else:
                loss_value = loss_value, {loss_func.name: loss_value}
            cum_loss = cum_loss + weight * loss_value[0]
            for key, val in loss_value[1].items():
                details["loss_" + key] = val
        details["loss"] = cum_loss
        return cum_loss.mean(), details
