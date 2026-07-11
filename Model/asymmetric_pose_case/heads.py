from torch import nn

from .embeddings import build_mlp


class PoseDeltaPredictionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.head = build_mlp(input_dim, hidden_dims, output_dim, dropout)

    def forward(self, fused_features):
        return self.head(fused_features)
