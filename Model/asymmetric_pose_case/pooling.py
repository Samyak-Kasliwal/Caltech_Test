import torch
from torch import nn

from .config import PoolingConfig, SequenceConfig


class LastValidStepPooling(nn.Module):
    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        indices = (lengths - 1).clamp(min=0)
        batch_indices = torch.arange(sequence.shape[0], device=sequence.device)
        return sequence[batch_indices, indices]


class MaskMeanPooling(nn.Module):
    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        time_steps = sequence.shape[1]
        mask = torch.arange(time_steps, device=sequence.device)[None, :] < lengths[:, None]
        masked = sequence * mask.unsqueeze(-1)
        return masked.sum(dim=1) / lengths.clamp(min=1).unsqueeze(-1)


class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int, config: PoolingConfig) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, config.attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(config.attention_hidden_dim, 1),
        )

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        time_steps = sequence.shape[1]
        mask = torch.arange(time_steps, device=sequence.device)[None, :] < lengths[:, None]
        scores = self.scorer(sequence).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(sequence * weights.unsqueeze(-1), dim=1)


def build_pooler(
    pooling_config: PoolingConfig, sequence_config: SequenceConfig, input_dim: int
) -> nn.Module:
    pooling_type = pooling_config.pooling_type.lower()
    if pooling_type == "auto":
        pooling_type = "last" if sequence_config.model_type.lower() == "lstm" else "mean"
    if pooling_type == "last":
        return LastValidStepPooling()
    if pooling_type == "mean":
        return MaskMeanPooling()
    if pooling_type == "attention":
        return AttentionPooling(input_dim, pooling_config)
    raise ValueError(f"Unsupported pooling_type: {pooling_config.pooling_type}")
