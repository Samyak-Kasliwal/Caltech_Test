from torch import nn


def build_loss(loss_name: str = "mse") -> nn.Module:
    loss_name = loss_name.lower()
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name in {"smooth_l1", "huber"}:
        return nn.SmoothL1Loss()
    raise ValueError(f"Unsupported loss: {loss_name}")
