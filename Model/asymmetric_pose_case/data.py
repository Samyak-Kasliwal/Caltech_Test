from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DataConfig, NormalizationConfig


NOSE_INDEX = 0
NECK_INDEX = 3


@dataclass(frozen=True)
class SequenceRecord:
    group_id: str
    sequence_id: str
    keypoints: np.ndarray
    velocity: np.ndarray
    annotations: np.ndarray


@dataclass(frozen=True)
class WindowIndex:
    sequence_index: int
    target_t: int


@dataclass
class FeatureNormalizer:
    mean: torch.Tensor
    std: torch.Tensor

    def transform(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.mean.to(value.device)) / self.std.to(value.device)

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.std.to(value.device) + self.mean.to(value.device)


@dataclass
class NormalizerBundle:
    coord: FeatureNormalizer
    velocity: FeatureNormalizer
    interaction: FeatureNormalizer
    target_delta: FeatureNormalizer


class OnlineStats:
    def __init__(self, feature_dim: int) -> None:
        self.count = 0
        self.sum = np.zeros(feature_dim, dtype=np.float64)
        self.sum_sq = np.zeros(feature_dim, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1, self.sum.shape[0])
        if values.size == 0:
            return
        self.count += values.shape[0]
        self.sum += values.sum(axis=0)
        self.sum_sq += np.square(values).sum(axis=0)

    def finalize(self, eps: float) -> FeatureNormalizer:
        if self.count == 0:
            raise ValueError("Cannot finalize normalizer with zero observations.")
        mean = self.sum / self.count
        var = np.maximum(self.sum_sq / self.count - np.square(mean), 0.0)
        std = np.sqrt(var) + eps
        return FeatureNormalizer(
            mean=torch.tensor(mean, dtype=torch.float32),
            std=torch.tensor(std, dtype=torch.float32),
        )


def load_calms21_sequences(data_path: Path) -> list[SequenceRecord]:
    raw_data = np.load(data_path, allow_pickle=True).item()
    records: list[SequenceRecord] = []

    for group_id, sequences in raw_data.items():
        for sequence_id, sequence_data in sequences.items():
            keypoints = np.asarray(sequence_data["keypoints"], dtype=np.float32)
            annotations = np.asarray(sequence_data["annotations"], dtype=np.int64)
            if len(keypoints) != len(annotations):
                raise ValueError(f"{sequence_id} has mismatched keypoints/annotations.")
            records.append(
                SequenceRecord(
                    group_id=group_id,
                    sequence_id=sequence_id,
                    keypoints=keypoints,
                    velocity=compute_velocity(keypoints),
                    annotations=annotations,
                )
            )

    return records


def build_window_indices(
    records: list[SequenceRecord], data_config: DataConfig
) -> list[WindowIndex]:
    windows: list[WindowIndex] = []
    start_t = data_config.min_target_index
    for sequence_index, record in enumerate(records):
        for target_t in range(start_t, len(record.annotations)):
            windows.append(WindowIndex(sequence_index=sequence_index, target_t=target_t))
    return windows


def keypoints_to_joint_xy(keypoints: np.ndarray, mouse_index: int) -> np.ndarray:
    # Convert CalMS21 [time, mouse, coord, joint] into [time, joint, coord].
    # 将 CalMS21 的 [时间, 小鼠, 坐标, 关节] 转换为 [时间, 关节, 坐标]。
    return np.transpose(keypoints[:, mouse_index, :, :], (0, 2, 1)).astype(np.float32)


def compute_velocity(keypoints: np.ndarray) -> np.ndarray:
    velocity = np.zeros_like(keypoints, dtype=np.float32)
    velocity[1:] = keypoints[1:] - keypoints[:-1]
    return velocity


def compute_interaction_features(keypoints: np.ndarray, frame_indices: np.ndarray) -> np.ndarray:
    joint_xy = np.stack(
        [keypoints_to_joint_xy(keypoints[frame_indices], 0), keypoints_to_joint_xy(keypoints[frame_indices], 1)],
        axis=1,
    )
    a_xy = joint_xy[:, 0]
    b_xy = joint_xy[:, 1]

    a_center = np.nanmean(a_xy, axis=1)
    b_center = np.nanmean(b_xy, axis=1)
    center_delta = b_center - a_center
    center_distance = np.linalg.norm(center_delta, axis=1, keepdims=True)

    a_heading = a_xy[:, NOSE_INDEX, :] - a_xy[:, NECK_INDEX, :]
    b_heading = b_xy[:, NOSE_INDEX, :] - b_xy[:, NECK_INDEX, :]
    a_norm = np.linalg.norm(a_heading, axis=1) + 1e-6
    b_norm = np.linalg.norm(b_heading, axis=1) + 1e-6
    dot = np.sum(a_heading * b_heading, axis=1, keepdims=True)
    cross = (
        a_heading[:, 0:1] * b_heading[:, 1:2]
        - a_heading[:, 1:2] * b_heading[:, 0:1]
    )
    angle_cos = dot / (a_norm[:, None] * b_norm[:, None])
    angle_sin = cross / (a_norm[:, None] * b_norm[:, None])

    # Use sin/cos for the facing angle to avoid angle wrap-around discontinuity.
    # 使用 sin/cos 表示面朝方向夹角，避免角度在正负 pi 附近跳变。
    return np.concatenate(
        [center_delta, center_distance, angle_sin, angle_cos], axis=1
    ).astype(np.float32)


def behavior_labels_for_indices(
    annotations: np.ndarray, frame_indices: np.ndarray, mode: str
) -> np.ndarray:
    if mode == "history_plus_current":
        return annotations[frame_indices]
    if mode == "history_only":
        return annotations[np.maximum(frame_indices - 1, 0)]
    if mode == "none":
        return np.zeros(len(frame_indices), dtype=np.int64)
    raise ValueError(f"Unknown behavior_label_mode: {mode}")


def extract_window_arrays(
    record: SequenceRecord, window: WindowIndex, data_config: DataConfig
) -> dict[str, np.ndarray | np.int64 | str]:
    t = window.target_t
    keypoints = record.keypoints
    velocity = record.velocity

    a_indices = np.arange(t - data_config.a_history_frames, t + 1)
    b_indices = np.arange(t - data_config.b_history_frames, t)
    interaction_indices = b_indices

    a_xy = keypoints_to_joint_xy(keypoints[a_indices], 0)
    b_xy = keypoints_to_joint_xy(keypoints[b_indices], 1)
    a_velocity = keypoints_to_joint_xy(velocity[a_indices], 0)
    b_velocity = keypoints_to_joint_xy(velocity[b_indices], 1)

    # B velocity at t-1 is B[t-1] - B[t-2], so target B[t] never leaks in.
    # B 在 t-1 的速度定义为 B[t-1] - B[t-2]，不会泄露目标帧 B[t]。
    interaction = compute_interaction_features(keypoints, interaction_indices)
    target_delta = (
        keypoints_to_joint_xy(keypoints[[t]], 1)[0]
        - keypoints_to_joint_xy(keypoints[[t - 1]], 1)[0]
    ).reshape(-1)
    b_previous_pose = keypoints_to_joint_xy(keypoints[[t - 1]], 1)[0].reshape(-1)
    target_pose = keypoints_to_joint_xy(keypoints[[t]], 1)[0].reshape(-1)

    return {
        "a_xy": a_xy,
        "b_xy": b_xy,
        "a_velocity": a_velocity,
        "b_velocity": b_velocity,
        "interaction": interaction,
        "a_behavior": behavior_labels_for_indices(
            record.annotations, a_indices, data_config.behavior_label_mode
        ),
        "b_behavior": behavior_labels_for_indices(
            record.annotations, b_indices, data_config.behavior_label_mode
        ),
        "current_behavior": np.int64(record.annotations[t]),
        "target_delta": target_delta.astype(np.float32),
        "b_previous_pose": b_previous_pose.astype(np.float32),
        "target_pose": target_pose.astype(np.float32),
        "target_t": np.int64(t),
        "sequence_id": record.sequence_id,
    }


def fit_normalizers(
    records: list[SequenceRecord],
    windows: list[WindowIndex],
    data_config: DataConfig,
    normalization_config: NormalizationConfig,
) -> NormalizerBundle:
    coord_stats = OnlineStats(data_config.coord_dim)
    velocity_stats = OnlineStats(data_config.coord_dim)
    interaction_stats = OnlineStats(5)
    target_stats = OnlineStats(data_config.num_joints * data_config.coord_dim)

    for window in windows:
        record = records[window.sequence_index]
        arrays = extract_window_arrays(record, window, data_config)
        coord_stats.update(arrays["a_xy"].reshape(-1, 2))
        coord_stats.update(arrays["b_xy"].reshape(-1, 2))
        velocity_stats.update(arrays["a_velocity"].reshape(-1, 2))
        velocity_stats.update(arrays["b_velocity"].reshape(-1, 2))
        interaction_stats.update(arrays["interaction"])
        target_stats.update(arrays["target_delta"])

    return NormalizerBundle(
        coord=coord_stats.finalize(normalization_config.eps),
        velocity=velocity_stats.finalize(normalization_config.eps),
        interaction=interaction_stats.finalize(normalization_config.eps),
        target_delta=target_stats.finalize(normalization_config.eps),
    )


class CalMS21AsymmetricPoseDataset(Dataset):
    def __init__(
        self,
        records: list[SequenceRecord],
        windows: list[WindowIndex],
        data_config: DataConfig,
        normalizers: NormalizerBundle,
    ) -> None:
        self.records = records
        self.windows = windows
        self.data_config = data_config
        self.normalizers = normalizers

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        window = self.windows[index]
        record = self.records[window.sequence_index]
        arrays = extract_window_arrays(record, window, self.data_config)

        a_xy = self.normalizers.coord.transform(torch.tensor(arrays["a_xy"]))
        b_xy = self.normalizers.coord.transform(torch.tensor(arrays["b_xy"]))
        a_velocity = self.normalizers.velocity.transform(torch.tensor(arrays["a_velocity"]))
        b_velocity = self.normalizers.velocity.transform(torch.tensor(arrays["b_velocity"]))
        interaction = self.normalizers.interaction.transform(torch.tensor(arrays["interaction"]))
        target_delta = self.normalizers.target_delta.transform(
            torch.tensor(arrays["target_delta"])
        )

        return {
            "a_xy": a_xy,
            "b_xy": b_xy,
            "a_velocity": a_velocity,
            "b_velocity": b_velocity,
            "interaction": interaction,
            "a_behavior": torch.tensor(arrays["a_behavior"], dtype=torch.long),
            "b_behavior": torch.tensor(arrays["b_behavior"], dtype=torch.long),
            "current_behavior": torch.tensor(arrays["current_behavior"], dtype=torch.long),
            "target_delta": target_delta,
            "b_previous_pose": torch.tensor(arrays["b_previous_pose"], dtype=torch.float32),
            "target_pose": torch.tensor(arrays["target_pose"], dtype=torch.float32),
            "target_t": torch.tensor(arrays["target_t"], dtype=torch.long),
            "sequence_id": arrays["sequence_id"],
        }
