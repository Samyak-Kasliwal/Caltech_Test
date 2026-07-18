from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import default_config
from .data import FeatureNormalizer, NormalizerBundle
from .embeddings import AsymmetricEmbeddingModule
from .heads import PoseDeltaPredictionHead
from .pooling import build_pooler
from .sequence_models import build_sequence_model


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value


def config_from_dict(payload: dict[str, Any]):
    cfg = default_config()

    def apply_updates(target: Any, values: dict[str, Any]) -> None:
        for field_info in fields(target):
            if field_info.name not in values:
                continue
            current_value = getattr(target, field_info.name)
            incoming_value = values[field_info.name]
            if is_dataclass(current_value) and isinstance(incoming_value, dict):
                apply_updates(current_value, incoming_value)
            elif isinstance(current_value, Path):
                setattr(target, field_info.name, Path(incoming_value))
            elif isinstance(current_value, tuple) and isinstance(incoming_value, list):
                setattr(target, field_info.name, tuple(incoming_value))
            else:
                setattr(target, field_info.name, incoming_value)

    apply_updates(cfg, payload)
    return cfg


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "TrainingConfig.device is set to 'cuda', but torch.cuda.is_available() is False."
        )
    return torch.device(device_name)


def print_preflight(device: torch.device) -> None:
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Selected device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def make_lengths(batch_size: int, length: int, device: torch.device) -> torch.Tensor:
    return torch.full((batch_size,), length, dtype=torch.long, device=device)


def build_modules(cfg, device: torch.device) -> dict[str, nn.Module]:
    embedder = AsymmetricEmbeddingModule(cfg.data, cfg.embedding).to(device)

    # A/B share one temporal model instance so Single/Multi keep matched capacity.
    # A/B 共用同一个时序模型实例，使 Single/Multi 保持一致的时序建模容量。
    mouse_sequence = build_sequence_model(
        cfg.sequence.model_type,
        cfg.embedding.mouse_embedding_dim,
        cfg.sequence.hidden_dim,
        cfg.sequence,
    ).to(device)
    mouse_pooler = build_pooler(cfg.pooling, cfg.sequence, cfg.sequence.output_dim).to(device)

    current_behavior_dim = (
        cfg.embedding.behavior_embedding_dim
        if cfg.data.behavior_label_mode == "history_plus_current"
        else 0
    )
    model_name = cfg.data.model_name.lower()
    if model_name == "single_predict":
        interaction_sequence = None
        interaction_pooler = None
        head_input_dim = cfg.sequence.output_dim + current_behavior_dim
    elif model_name == "multi_predict":
        interaction_sequence = build_sequence_model(
            cfg.sequence.model_type,
            cfg.embedding.interaction_embedding_dim,
            cfg.sequence.interaction_hidden_dim,
            cfg.sequence,
        ).to(device)
        interaction_pooler = build_pooler(
            cfg.pooling, cfg.sequence, cfg.sequence.interaction_output_dim
        ).to(device)
        head_input_dim = (
            cfg.sequence.output_dim * 2
            + cfg.sequence.interaction_output_dim
            + current_behavior_dim
        )
    else:
        raise ValueError(f"Unsupported model_name: {cfg.data.model_name}")

    prediction_head = PoseDeltaPredictionHead(
        head_input_dim,
        cfg.head.hidden_dims,
        cfg.head.output_dim,
        cfg.head.dropout,
    ).to(device)

    return {
        "embedder": embedder,
        "mouse_sequence": mouse_sequence,
        "mouse_pooler": mouse_pooler,
        "prediction_head": prediction_head,
        **(
            {
                "interaction_sequence": interaction_sequence,
                "interaction_pooler": interaction_pooler,
            }
            if model_name == "multi_predict"
            else {}
        ),
    }


def forward_batch(
    batch: dict[str, Any],
    modules: dict[str, nn.Module],
    cfg,
    device: torch.device,
) -> torch.Tensor:
    batch_size = batch["a_xy"].shape[0]

    # Encode raw coordinates, velocities, and labels into latent tokens.
    # 将原始坐标、速度和标签编码为 latent token。
    embedded = modules["embedder"](batch)

    # Run temporal modeling over the target A branch.
    # 对目标 A 分支执行时序建模。
    a_sequence = modules["mouse_sequence"](embedded["a"])

    a_lengths = make_lengths(batch_size, cfg.data.a_window_length, device)

    # Pool temporal features into one fixed-size vector.
    # 将时序特征聚合为固定维度向量。
    a_pooled = modules["mouse_pooler"](a_sequence, a_lengths)

    fused_features = [a_pooled]
    if cfg.data.model_name.lower() == "multi_predict":
        # Multi_predict additionally uses B context and interaction context.
        # Multi_predict 额外使用 B 分支上下文和 interaction 上下文。
        b_sequence = modules["mouse_sequence"](embedded["b"])
        interaction_sequence = modules["interaction_sequence"](embedded["interaction"])
        b_lengths = make_lengths(batch_size, cfg.data.b_window_length, device)
        interaction_lengths = make_lengths(
            batch_size, cfg.data.interaction_window_length, device
        )
        b_pooled = modules["mouse_pooler"](b_sequence, b_lengths)
        interaction_pooled = modules["interaction_pooler"](
            interaction_sequence, interaction_lengths
        )
        fused_features.extend([b_pooled, interaction_pooled])

    if cfg.data.behavior_label_mode == "history_plus_current":
        # Current annotation is a global condition, not an A/B-specific label.
        # 当前 annotation 是全局条件，不属于任意一只小鼠的私有标签。
        current_behavior = modules["embedder"].embed_behavior(batch["current_behavior"])
        fused_features.append(current_behavior)

    # Predict target mouse current pose displacement A[t] - A[t-1].
    # 预测目标鼠当前姿态相对上一帧的位移 A[t] - A[t-1]。
    return modules["prediction_head"](torch.cat(fused_features, dim=-1))


def normalizers_state_dict(normalizers) -> dict[str, dict[str, list[float]]]:
    return {
        name: {
            "mean": getattr(normalizers, name).mean.cpu().tolist(),
            "std": getattr(normalizers, name).std.cpu().tolist(),
        }
        for name in ("coord", "velocity", "self_distance", "interaction", "target_delta")
    }


def normalizers_from_state_dict(state: dict[str, dict[str, list[float]]]) -> NormalizerBundle:
    def build(name: str) -> FeatureNormalizer:
        return FeatureNormalizer(
            mean=torch.tensor(state[name]["mean"], dtype=torch.float32),
            std=torch.tensor(state[name]["std"], dtype=torch.float32),
        )

    return NormalizerBundle(
        coord=build("coord"),
        velocity=build("velocity"),
        self_distance=build("self_distance"),
        interaction=build("interaction"),
        target_delta=build("target_delta"),
    )


def save_checkpoint(
    path: Path,
    modules: dict[str, nn.Module],
    optimizer: torch.optim.Optimizer,
    cfg,
    normalizers,
    fold_index: int,
    epoch: int,
    train_loss: float,
    val_loss: float,
) -> None:
    checkpoint = {
        "fold": fold_index,
        "epoch": epoch,
        "train_mse": train_loss,
        "val_mse": val_loss,
        "config": to_jsonable(cfg),
        "normalizers": normalizers_state_dict(normalizers),
        "modules": {name: module.state_dict() for name, module in modules.items()},
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)


def load_checkpoint(checkpoint_path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(checkpoint_path, map_location=map_location)


def load_module_states(
    modules: dict[str, nn.Module], checkpoint: dict[str, Any] | Path
) -> None:
    if isinstance(checkpoint, Path):
        checkpoint = load_checkpoint(checkpoint)
    for name, module in modules.items():
        module.load_state_dict(checkpoint["modules"][name])


def build_modules_from_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> tuple[dict[str, nn.Module], NormalizerBundle, dict[str, Any]]:
    # Load checkpoint once, then rebuild modules with the current config schema.
    # 只加载一次 checkpoint，再用当前配置结构重建模型模块。
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    cfg = config_from_dict(checkpoint["config"])
    modules = build_modules(cfg, device)
    load_module_states(modules, checkpoint)
    normalizers = normalizers_from_state_dict(checkpoint["normalizers"])
    return modules, normalizers, checkpoint
