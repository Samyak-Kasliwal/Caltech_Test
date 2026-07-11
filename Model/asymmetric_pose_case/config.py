from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    data_path: Path = Path("Caltech/calms21_task1_train_distance_lt_330.npy")
    num_mice: int = 2
    num_joints: int = 7
    coord_dim: int = 2
    num_behavior_classes: int = 4
    a_history_frames: int = 9
    a_include_current: bool = True
    b_history_frames: int = 4
    behavior_label_mode: str = "history_plus_current"
    use_scores: bool = False

    @property
    def a_window_length(self) -> int:
        return self.a_history_frames + int(self.a_include_current)

    @property
    def b_window_length(self) -> int:
        return self.b_history_frames

    @property
    def min_target_index(self) -> int:
        return max(self.a_history_frames, self.b_history_frames)


@dataclass
class EmbeddingConfig:
    joint_input_dim: int = 2
    joint_embedding_dim: int = 16
    pose_embedding_dim: int = 64
    velocity_embedding_dim: int = 32
    behavior_embedding_dim: int = 16
    mouse_embedding_dim: int = 128
    interaction_input_dim: int = 5
    interaction_embedding_dim: int = 32
    frame_mlp_hidden_dim: int = 128
    dropout: float = 0.0


@dataclass
class SequenceConfig:
    model_type: str = "lstm"
    input_dim: int = 128
    interaction_input_dim: int = 32
    hidden_dim: int = 128
    interaction_hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    bidirectional: bool = False

    @property
    def output_dim(self) -> int:
        return self.hidden_dim * (2 if self.bidirectional else 1)

    @property
    def interaction_output_dim(self) -> int:
        return self.interaction_hidden_dim * (2 if self.bidirectional else 1)


@dataclass
class PoolingConfig:
    pooling_type: str = "auto"
    attention_hidden_dim: int = 64


@dataclass
class HeadConfig:
    hidden_dims: tuple[int, ...] = (256, 128)
    output_dim: int = 14
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    num_folds: int = 5
    split_mode: str = "sequence_level"
    seed: int = 42
    batch_size: int = 256
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    device: str = "auto"
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    output_root: Path = Path("Model/asymmetric_pose_case/runs")
    experiment_name: str = "asymmetric_pose"
    save_checkpoints: bool = True
    save_best_checkpoint: bool = True
    save_last_checkpoint: bool = True
    export_validation_predictions: bool = True


@dataclass
class NormalizationConfig:
    eps: float = 1e-6


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    pooling: PoolingConfig = field(default_factory=PoolingConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)


def default_config() -> ExperimentConfig:
    return ExperimentConfig()
