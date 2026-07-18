import torch
from torch import nn

from .config import DataConfig, EmbeddingConfig


def build_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    dropout: float = 0.0,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend([nn.Linear(last_dim, hidden_dim), nn.ReLU()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class JointEncoder(nn.Module):
    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__()
        self.encoder = build_mlp(
            config.joint_input_dim,
            (config.joint_embedding_dim,),
            config.joint_embedding_dim,
            config.dropout,
        )

    def forward(self, joint_xy: torch.Tensor) -> torch.Tensor:
        return self.encoder(joint_xy)


class PoseEncoder(nn.Module):
    def __init__(
        self, data_config: DataConfig, config: EmbeddingConfig, joint_encoder: JointEncoder
    ) -> None:
        super().__init__()
        self.joint_encoder = joint_encoder
        self.pose_mlp = build_mlp(
            data_config.num_joints * config.joint_embedding_dim,
            (config.frame_mlp_hidden_dim,),
            config.pose_embedding_dim,
            config.dropout,
        )

    def forward(self, joint_xy: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, joint_count, coord_dim = joint_xy.shape
        joint_latent = self.joint_encoder(joint_xy.reshape(-1, coord_dim))
        joint_latent = joint_latent.reshape(batch_size, time_steps, joint_count, -1)
        return self.pose_mlp(joint_latent.reshape(batch_size, time_steps, -1))


class VelocityEncoder(nn.Module):
    def __init__(self, data_config: DataConfig, config: EmbeddingConfig) -> None:
        super().__init__()
        self.velocity_joint_encoder = JointEncoder(config)
        self.velocity_mlp = build_mlp(
            data_config.num_joints * config.joint_embedding_dim,
            (config.frame_mlp_hidden_dim,),
            config.velocity_embedding_dim,
            config.dropout,
        )

    def forward(self, joint_velocity: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, joint_count, coord_dim = joint_velocity.shape
        velocity_latent = self.velocity_joint_encoder(joint_velocity.reshape(-1, coord_dim))
        velocity_latent = velocity_latent.reshape(batch_size, time_steps, joint_count, -1)
        return self.velocity_mlp(velocity_latent.reshape(batch_size, time_steps, -1))


class BehaviorEncoder(nn.Module):
    def __init__(self, data_config: DataConfig, embedding_config: EmbeddingConfig) -> None:
        super().__init__()
        self.enabled = data_config.behavior_label_mode != "none"
        self.embedding_dim = embedding_config.behavior_embedding_dim
        self.embedding = nn.Embedding(
            data_config.num_behavior_classes, embedding_config.behavior_embedding_dim
        )

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            return self.embedding(labels)
        return torch.zeros(*labels.shape, self.embedding_dim, device=labels.device)


class RoleEncoder(nn.Module):
    def __init__(self, data_config: DataConfig, embedding_config: EmbeddingConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            data_config.num_role_classes, embedding_config.role_embedding_dim
        )

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding(labels)


class SelfDistanceEncoder(nn.Module):
    def __init__(self, data_config: DataConfig, embedding_config: EmbeddingConfig) -> None:
        super().__init__()
        self.encoder = build_mlp(
            data_config.num_self_distances,
            (embedding_config.frame_mlp_hidden_dim,),
            embedding_config.self_distance_embedding_dim,
            embedding_config.dropout,
        )

    def forward(self, self_distances: torch.Tensor) -> torch.Tensor:
        return self.encoder(self_distances)


class MouseEncoder(nn.Module):
    def __init__(self, data_config: DataConfig, embedding_config: EmbeddingConfig) -> None:
        super().__init__()
        shared_joint_encoder = JointEncoder(embedding_config)
        self.pose_encoder = PoseEncoder(data_config, embedding_config, shared_joint_encoder)
        self.velocity_encoder = VelocityEncoder(data_config, embedding_config)
        self.self_distance_encoder = SelfDistanceEncoder(data_config, embedding_config)
        self.behavior_encoder = BehaviorEncoder(data_config, embedding_config)
        self.role_encoder = RoleEncoder(data_config, embedding_config)

        mouse_input_dim = (
            embedding_config.pose_embedding_dim
            + embedding_config.velocity_embedding_dim
            + embedding_config.self_distance_embedding_dim
            + embedding_config.behavior_embedding_dim
            + embedding_config.role_embedding_dim
        )
        self.mouse_mlp = build_mlp(
            mouse_input_dim,
            (embedding_config.frame_mlp_hidden_dim,),
            embedding_config.mouse_embedding_dim,
            embedding_config.dropout,
        )

    def forward(
        self,
        joint_xy: torch.Tensor,
        joint_velocity: torch.Tensor,
        self_distances: torch.Tensor,
        behavior_labels: torch.Tensor,
        role_labels: torch.Tensor,
    ) -> torch.Tensor:
        # Encode each semantic modality before mouse-level fusion.
        # 先分别编码每一种语义模态，再做 mouse-level 融合。
        pose_embedding = self.pose_encoder(joint_xy)
        velocity_embedding = self.velocity_encoder(joint_velocity)
        self_distance_embedding = self.self_distance_encoder(self_distances)
        behavior_embedding = self.behavior_encoder(behavior_labels)
        role_embedding = self.role_encoder(role_labels)
        mouse_features = torch.cat(
            [
                pose_embedding,
                velocity_embedding,
                self_distance_embedding,
                behavior_embedding,
                role_embedding,
            ],
            dim=-1,
        )
        return self.mouse_mlp(mouse_features)


class InteractionEncoder(nn.Module):
    def __init__(self, embedding_config: EmbeddingConfig) -> None:
        super().__init__()
        self.encoder = build_mlp(
            embedding_config.interaction_input_dim,
            (embedding_config.frame_mlp_hidden_dim,),
            embedding_config.interaction_embedding_dim,
            embedding_config.dropout,
        )

    def forward(self, interaction_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(interaction_features)


class AsymmetricEmbeddingModule(nn.Module):
    def __init__(self, data_config: DataConfig, embedding_config: EmbeddingConfig) -> None:
        super().__init__()
        self.mouse_encoder = MouseEncoder(data_config, embedding_config)
        self.interaction_encoder = InteractionEncoder(embedding_config)

    def embed_behavior(self, labels: torch.Tensor) -> torch.Tensor:
        return self.mouse_encoder.behavior_encoder(labels)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # A/B share the same mouse encoder so both animals live in one latent space.
        # A/B 共用同一个 mouse encoder，确保两只小鼠处在同一个 latent 空间。
        a_embedding = self.mouse_encoder(
            batch["a_xy"],
            batch["a_velocity"],
            batch["a_self_distance"],
            batch["a_behavior"],
            batch["a_role"],
        )
        b_embedding = self.mouse_encoder(
            batch["b_xy"],
            batch["b_velocity"],
            batch["b_self_distance"],
            batch["b_behavior"],
            batch["b_role"],
        )

        # Interaction features are encoded as an independent modality.
        # 交互特征作为独立模态编码，不提前合并到任意一只小鼠中。
        interaction_embedding = self.interaction_encoder(batch["interaction"])
        return {
            "a": a_embedding,
            "b": b_embedding,
            "interaction": interaction_embedding,
        }
