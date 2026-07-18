from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DataConfig, NormalizationConfig


NODE_NAMES = ["head", "headLeft", "headRight", "neck", "bodyLeft", "bodyRight", "tail"]
BRANCH_INDEX = {"intruder": 0, "resident": 1}
CACHE_PREPROCESS_VERSION = 1

FLOAT_FEATURE_KEYS = (
    "a_xy",
    "a_velocity",
    "a_self_distance",
    "b_xy",
    "b_velocity",
    "b_self_distance",
    "interaction",
    "target_delta",
    "previous_pose",
    "target_pose",
)
INT_FEATURE_KEYS = (
    "a_behavior",
    "a_role",
    "b_behavior",
    "b_role",
    "current_behavior",
    "target_t",
    "sequence_index",
    "window_index",
)


@dataclass(frozen=True)
class SequenceRecord:
    sequence_id: str
    windows: list[list[list[dict[str, Any]]]]


@dataclass(frozen=True)
class WindowIndex:
    sequence_index: int
    window_index: int
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
    self_distance: FeatureNormalizer
    interaction: FeatureNormalizer
    target_delta: FeatureNormalizer


@dataclass
class WindowFeatureCache:
    path: Path
    arrays: dict[str, np.ndarray]
    sequence_ids: list[str]
    manifest: dict[str, Any]
    status: str

    def __post_init__(self) -> None:
        self._window_to_row = {
            (
                int(sequence_index),
                int(window_index),
                int(target_t),
            ): row
            for row, (sequence_index, window_index, target_t) in enumerate(
                zip(
                    self.arrays["sequence_index"],
                    self.arrays["window_index"],
                    self.arrays["target_t"],
                )
            )
        }

    def __len__(self) -> int:
        return int(self.manifest["num_windows"])

    def row_indices(self, windows: list["WindowIndex"]) -> np.ndarray:
        return np.asarray(
            [
                self._window_to_row[
                    (window.sequence_index, window.window_index, window.target_t)
                ]
                for window in windows
            ],
            dtype=np.int64,
        )


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
            mean = np.zeros_like(self.sum)
            std = np.ones_like(self.sum)
        else:
            mean = self.sum / self.count
            var = np.maximum(self.sum_sq / self.count - np.square(mean), 0.0)
            std = np.sqrt(var) + eps
        return FeatureNormalizer(
            mean=torch.tensor(mean, dtype=torch.float32),
            std=torch.tensor(std, dtype=torch.float32),
        )


def load_calms21_sequences(data_path: Path) -> list[SequenceRecord]:
    raw_data = np.load(data_path, allow_pickle=True).item()
    return [
        SequenceRecord(sequence_id=sequence_id, windows=windows)
        for sequence_id, windows in raw_data.items()
    ]


def build_window_indices(
    records: list[SequenceRecord], data_config: DataConfig
) -> list[WindowIndex]:
    windows: list[WindowIndex] = []
    start_t = data_config.min_target_index
    for sequence_index, record in enumerate(records):
        for window_index, window in enumerate(record.windows):
            target_frames = get_branch_frames(window, data_config.target_branch)
            for target_t in range(start_t, len(target_frames)):
                windows.append(
                    WindowIndex(
                        sequence_index=sequence_index,
                        window_index=window_index,
                        target_t=target_t,
                    )
                )
    return windows


def get_branch_frames(
    window: list[list[dict[str, Any]]], branch_name: str
) -> list[dict[str, Any]]:
    try:
        return window[BRANCH_INDEX[branch_name]]
    except KeyError as exc:
        raise ValueError(f"Unknown branch_name: {branch_name}") from exc


def frame_position(frame: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [frame["body"]["node"][node_name]["position"] for node_name in NODE_NAMES],
        dtype=np.float32,
    )


def frame_velocity(frame: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [frame["body"]["node"][node_name]["node_velocity"] for node_name in NODE_NAMES],
        dtype=np.float32,
    )


def frame_self_distance(frame: dict[str, Any]) -> np.ndarray:
    return np.asarray(frame["body"]["internode_distance"], dtype=np.float32)


def frame_annotation_id(frame: dict[str, Any]) -> np.int64:
    tag = np.asarray(frame["interaction"]["annotation_tag"], dtype=np.float32)
    return np.int64(np.argmax(tag))


def frame_role_label(frame: dict[str, Any]) -> np.int64:
    # Role labels are read from preprocessed data; 0/1 are category values, not mouse ids.
    # 身份标签直接读取预处理数据；0/1 是类别值，不是老鼠编号。
    return np.int64(frame["interaction"]["intruder_or_resident_tag"])


def branch_arrays(frames: list[dict[str, Any]], indices: np.ndarray) -> dict[str, np.ndarray]:
    selected = [frames[int(index)] for index in indices]
    return {
        "xy": np.stack([frame_position(frame) for frame in selected]).astype(np.float32),
        "velocity": np.stack([frame_velocity(frame) for frame in selected]).astype(np.float32),
        "self_distance": np.stack(
            [frame_self_distance(frame) for frame in selected]
        ).astype(np.float32),
        "annotation": np.asarray(
            [frame_annotation_id(frame) for frame in selected], dtype=np.int64
        ),
        "role": np.asarray([frame_role_label(frame) for frame in selected], dtype=np.int64),
    }


def mouse_center(joint_xy: np.ndarray) -> np.ndarray:
    return np.nanmean(joint_xy, axis=0)


def interaction_arrays(
    target_frames: list[dict[str, Any]],
    context_frames: list[dict[str, Any]],
    indices: np.ndarray,
) -> np.ndarray:
    features = []
    for index in indices:
        target_frame = target_frames[int(index)]
        context_frame = context_frames[int(index)]
        target_xy = frame_position(target_frame)
        context_xy = frame_position(context_frame)
        center_delta = mouse_center(context_xy) - mouse_center(target_xy)
        midpoint_distance = float(
            target_frame["interaction"]["mouse_midpoint_distance"]
        )
        target_angle = np.deg2rad(float(target_frame["interaction"]["head_angle"]))
        context_angle = np.deg2rad(float(context_frame["interaction"]["head_angle"]))
        features.append(
            [
                center_delta[0],
                center_delta[1],
                midpoint_distance,
                np.sin(target_angle),
                np.cos(target_angle),
                np.sin(context_angle),
                np.cos(context_angle),
            ]
        )
    return np.asarray(features, dtype=np.float32)


def extract_window_arrays(
    record: SequenceRecord, window: WindowIndex, data_config: DataConfig
) -> dict[str, np.ndarray | np.int64 | str]:
    trial_window = record.windows[window.window_index]
    target_frames = get_branch_frames(trial_window, data_config.target_branch)
    context_frames = get_branch_frames(trial_window, data_config.context_branch)
    t = window.target_t

    history_indices = np.arange(t - data_config.history_frames, t)
    context_end = t + int(data_config.include_context_current)
    context_indices = np.arange(t - data_config.history_frames, context_end)
    interaction_end = t + int(data_config.include_interaction_current)
    interaction_indices = np.arange(t - data_config.history_frames, interaction_end)

    a_arrays = branch_arrays(target_frames, history_indices)
    b_arrays = branch_arrays(context_frames, context_indices)
    interaction = interaction_arrays(target_frames, context_frames, interaction_indices)

    target_pose = frame_position(target_frames[t]).reshape(-1)
    previous_pose = frame_position(target_frames[t - 1]).reshape(-1)
    target_delta = target_pose - previous_pose

    # Current interaction uses target-frame information by design for conditional prediction.
    # 当前 interaction 按设计会使用目标帧信息，属于条件预测设定。
    return {
        "a_xy": a_arrays["xy"],
        "a_velocity": a_arrays["velocity"],
        "a_self_distance": a_arrays["self_distance"],
        "a_behavior": a_arrays["annotation"],
        "a_role": a_arrays["role"],
        "b_xy": b_arrays["xy"],
        "b_velocity": b_arrays["velocity"],
        "b_self_distance": b_arrays["self_distance"],
        "b_behavior": b_arrays["annotation"],
        "b_role": b_arrays["role"],
        "interaction": interaction,
        "current_behavior": frame_annotation_id(target_frames[t]),
        "target_delta": target_delta.astype(np.float32),
        "previous_pose": previous_pose.astype(np.float32),
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
    self_distance_stats = OnlineStats(data_config.num_self_distances)
    interaction_stats = OnlineStats(7)
    target_stats = OnlineStats(data_config.num_joints * data_config.coord_dim)

    for window in windows:
        record = records[window.sequence_index]
        arrays = extract_window_arrays(record, window, data_config)
        coord_stats.update(arrays["a_xy"].reshape(-1, 2))
        coord_stats.update(arrays["b_xy"].reshape(-1, 2))
        velocity_stats.update(arrays["a_velocity"].reshape(-1, 2))
        velocity_stats.update(arrays["b_velocity"].reshape(-1, 2))
        self_distance_stats.update(arrays["a_self_distance"])
        self_distance_stats.update(arrays["b_self_distance"])
        interaction_stats.update(arrays["interaction"])
        target_stats.update(arrays["target_delta"])

    return NormalizerBundle(
        coord=coord_stats.finalize(normalization_config.eps),
        velocity=velocity_stats.finalize(normalization_config.eps),
        self_distance=self_distance_stats.finalize(normalization_config.eps),
        interaction=interaction_stats.finalize(normalization_config.eps),
        target_delta=target_stats.finalize(normalization_config.eps),
    )


def _cache_config_payload(data_config: DataConfig) -> dict[str, Any]:
    data_path = Path(data_config.data_path)
    stat = data_path.stat()
    return {
        "preprocess_version": CACHE_PREPROCESS_VERSION,
        "source_data_path": str(data_path),
        "source_data_path_resolved": str(data_path.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "data_config": {
            "target_branch": data_config.target_branch,
            "context_branch": data_config.context_branch,
            "history_frames": data_config.history_frames,
            "include_context_current": data_config.include_context_current,
            "include_interaction_current": data_config.include_interaction_current,
            "num_joints": data_config.num_joints,
            "coord_dim": data_config.coord_dim,
            "num_self_distances": data_config.num_self_distances,
        },
    }


def _cache_dir_for_config(data_config: DataConfig) -> tuple[Path, dict[str, Any]]:
    payload = _cache_config_payload(data_config)
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return Path(data_config.feature_cache_dir) / f"asymmetric_pose_{digest}", payload


def _array_shapes(num_windows: int, data_config: DataConfig) -> dict[str, tuple[int, ...]]:
    pose_shape = (num_windows, data_config.num_joints * data_config.coord_dim)
    return {
        "a_xy": (
            num_windows,
            data_config.a_window_length,
            data_config.num_joints,
            data_config.coord_dim,
        ),
        "a_velocity": (
            num_windows,
            data_config.a_window_length,
            data_config.num_joints,
            data_config.coord_dim,
        ),
        "a_self_distance": (
            num_windows,
            data_config.a_window_length,
            data_config.num_self_distances,
        ),
        "a_behavior": (num_windows, data_config.a_window_length),
        "a_role": (num_windows, data_config.a_window_length),
        "b_xy": (
            num_windows,
            data_config.b_window_length,
            data_config.num_joints,
            data_config.coord_dim,
        ),
        "b_velocity": (
            num_windows,
            data_config.b_window_length,
            data_config.num_joints,
            data_config.coord_dim,
        ),
        "b_self_distance": (
            num_windows,
            data_config.b_window_length,
            data_config.num_self_distances,
        ),
        "b_behavior": (num_windows, data_config.b_window_length),
        "b_role": (num_windows, data_config.b_window_length),
        "interaction": (num_windows, data_config.interaction_window_length, 7),
        "current_behavior": (num_windows,),
        "target_delta": pose_shape,
        "previous_pose": pose_shape,
        "target_pose": pose_shape,
        "target_t": (num_windows,),
        "sequence_index": (num_windows,),
        "window_index": (num_windows,),
    }


def _load_feature_cache(cache_dir: Path, expected_payload: dict[str, Any]) -> WindowFeatureCache | None:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    for key, expected_value in expected_payload.items():
        if manifest.get(key) != expected_value:
            return None

    arrays = {}
    for key in (*FLOAT_FEATURE_KEYS, *INT_FEATURE_KEYS):
        array_path = cache_dir / f"{key}.npy"
        if not array_path.exists():
            return None
        arrays[key] = np.load(array_path, mmap_mode="c")
    return WindowFeatureCache(
        path=cache_dir,
        arrays=arrays,
        sequence_ids=list(manifest["sequence_ids"]),
        manifest=manifest,
        status="hit",
    )


def _build_feature_cache(
    records: list[SequenceRecord],
    windows: list[WindowIndex],
    data_config: DataConfig,
    cache_dir: Path,
    payload: dict[str, Any],
) -> WindowFeatureCache:
    if not windows:
        raise ValueError("Cannot build feature cache with zero windows.")

    cache_root = cache_dir.parent
    cache_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = cache_root / f".{cache_dir.name}.tmp_{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    shapes = _array_shapes(len(windows), data_config)
    arrays: dict[str, np.memmap] = {}
    try:
        for key in FLOAT_FEATURE_KEYS:
            arrays[key] = np.lib.format.open_memmap(
                tmp_dir / f"{key}.npy",
                mode="w+",
                dtype=np.float32,
                shape=shapes[key],
            )
        for key in INT_FEATURE_KEYS:
            arrays[key] = np.lib.format.open_memmap(
                tmp_dir / f"{key}.npy",
                mode="w+",
                dtype=np.int64,
                shape=shapes[key],
            )

        for row, window in enumerate(windows):
            record = records[window.sequence_index]
            extracted = extract_window_arrays(record, window, data_config)
            for key in FLOAT_FEATURE_KEYS:
                arrays[key][row] = extracted[key]
            for key in INT_FEATURE_KEYS:
                if key == "sequence_index":
                    arrays[key][row] = window.sequence_index
                elif key == "window_index":
                    arrays[key][row] = window.window_index
                else:
                    arrays[key][row] = extracted[key]

        for array in arrays.values():
            array.flush()
        arrays.clear()
        del array
        gc.collect()

        manifest = {
            **payload,
            "num_windows": len(windows),
            "num_sequences": len(records),
            "sequence_ids": [record.sequence_id for record in records],
            "arrays": {
                key: {
                    "filename": f"{key}.npy",
                    "dtype": "float32" if key in FLOAT_FEATURE_KEYS else "int64",
                    "shape": list(shapes[key]),
                }
                for key in (*FLOAT_FEATURE_KEYS, *INT_FEATURE_KEYS)
            },
        }
        (tmp_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        tmp_dir.rename(cache_dir)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise

    loaded = _load_feature_cache(cache_dir, payload)
    if loaded is None:
        raise RuntimeError(f"Built feature cache could not be loaded: {cache_dir}")
    loaded.status = "built"
    return loaded


def build_or_load_window_feature_cache(
    records: list[SequenceRecord],
    windows: list[WindowIndex],
    data_config: DataConfig,
    rebuild: bool = False,
) -> WindowFeatureCache:
    cache_dir, payload = _cache_dir_for_config(data_config)
    if not rebuild:
        loaded = _load_feature_cache(cache_dir, payload)
        if loaded is not None:
            return loaded

    return _build_feature_cache(records, windows, data_config, cache_dir, payload)


def fit_normalizers_from_cache(
    cache: WindowFeatureCache,
    row_indices: np.ndarray,
    data_config: DataConfig,
    normalization_config: NormalizationConfig,
    chunk_size: int = 8192,
) -> NormalizerBundle:
    coord_stats = OnlineStats(data_config.coord_dim)
    velocity_stats = OnlineStats(data_config.coord_dim)
    self_distance_stats = OnlineStats(data_config.num_self_distances)
    interaction_stats = OnlineStats(7)
    target_stats = OnlineStats(data_config.num_joints * data_config.coord_dim)

    for start in range(0, len(row_indices), chunk_size):
        rows = row_indices[start : start + chunk_size]
        coord_stats.update(cache.arrays["a_xy"][rows].reshape(-1, data_config.coord_dim))
        coord_stats.update(cache.arrays["b_xy"][rows].reshape(-1, data_config.coord_dim))
        velocity_stats.update(
            cache.arrays["a_velocity"][rows].reshape(-1, data_config.coord_dim)
        )
        velocity_stats.update(
            cache.arrays["b_velocity"][rows].reshape(-1, data_config.coord_dim)
        )
        self_distance_stats.update(
            cache.arrays["a_self_distance"][rows].reshape(
                -1, data_config.num_self_distances
            )
        )
        self_distance_stats.update(
            cache.arrays["b_self_distance"][rows].reshape(
                -1, data_config.num_self_distances
            )
        )
        interaction_stats.update(cache.arrays["interaction"][rows].reshape(-1, 7))
        target_stats.update(cache.arrays["target_delta"][rows])

    return NormalizerBundle(
        coord=coord_stats.finalize(normalization_config.eps),
        velocity=velocity_stats.finalize(normalization_config.eps),
        self_distance=self_distance_stats.finalize(normalization_config.eps),
        interaction=interaction_stats.finalize(normalization_config.eps),
        target_delta=target_stats.finalize(normalization_config.eps),
    )


def _normalizer_transform(
    normalizer: FeatureNormalizer, value: np.ndarray
) -> torch.Tensor:
    return normalizer.transform(torch.as_tensor(value, dtype=torch.float32))


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
        a_self_distance = self.normalizers.self_distance.transform(
            torch.tensor(arrays["a_self_distance"])
        )
        b_self_distance = self.normalizers.self_distance.transform(
            torch.tensor(arrays["b_self_distance"])
        )
        interaction = self.normalizers.interaction.transform(torch.tensor(arrays["interaction"]))
        target_delta = self.normalizers.target_delta.transform(
            torch.tensor(arrays["target_delta"])
        )

        return {
            "a_xy": a_xy,
            "a_velocity": a_velocity,
            "a_self_distance": a_self_distance,
            "a_behavior": torch.tensor(arrays["a_behavior"], dtype=torch.long),
            "a_role": torch.tensor(arrays["a_role"], dtype=torch.long),
            "b_xy": b_xy,
            "b_velocity": b_velocity,
            "b_self_distance": b_self_distance,
            "b_behavior": torch.tensor(arrays["b_behavior"], dtype=torch.long),
            "b_role": torch.tensor(arrays["b_role"], dtype=torch.long),
            "interaction": interaction,
            "current_behavior": torch.tensor(arrays["current_behavior"], dtype=torch.long),
            "target_delta": target_delta,
            "previous_pose": torch.tensor(arrays["previous_pose"], dtype=torch.float32),
            "target_pose": torch.tensor(arrays["target_pose"], dtype=torch.float32),
            "target_t": torch.tensor(arrays["target_t"], dtype=torch.long),
            "sequence_id": arrays["sequence_id"],
        }


class CachedCalMS21AsymmetricPoseDataset(Dataset):
    def __init__(
        self,
        cache: WindowFeatureCache,
        row_indices: np.ndarray,
        normalizers: NormalizerBundle,
    ) -> None:
        self.cache = cache
        self.row_indices = np.asarray(row_indices, dtype=np.int64)
        self.normalizers = normalizers

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = int(self.row_indices[index])
        arrays = self.cache.arrays
        sequence_index = int(arrays["sequence_index"][row])

        return {
            "a_xy": _normalizer_transform(self.normalizers.coord, arrays["a_xy"][row]),
            "a_velocity": _normalizer_transform(
                self.normalizers.velocity, arrays["a_velocity"][row]
            ),
            "a_self_distance": _normalizer_transform(
                self.normalizers.self_distance, arrays["a_self_distance"][row]
            ),
            "a_behavior": torch.as_tensor(arrays["a_behavior"][row], dtype=torch.long),
            "a_role": torch.as_tensor(arrays["a_role"][row], dtype=torch.long),
            "b_xy": _normalizer_transform(self.normalizers.coord, arrays["b_xy"][row]),
            "b_velocity": _normalizer_transform(
                self.normalizers.velocity, arrays["b_velocity"][row]
            ),
            "b_self_distance": _normalizer_transform(
                self.normalizers.self_distance, arrays["b_self_distance"][row]
            ),
            "b_behavior": torch.as_tensor(arrays["b_behavior"][row], dtype=torch.long),
            "b_role": torch.as_tensor(arrays["b_role"][row], dtype=torch.long),
            "interaction": _normalizer_transform(
                self.normalizers.interaction, arrays["interaction"][row]
            ),
            "current_behavior": torch.as_tensor(
                arrays["current_behavior"][row], dtype=torch.long
            ),
            "target_delta": _normalizer_transform(
                self.normalizers.target_delta, arrays["target_delta"][row]
            ),
            "previous_pose": torch.as_tensor(
                arrays["previous_pose"][row], dtype=torch.float32
            ),
            "target_pose": torch.as_tensor(arrays["target_pose"][row], dtype=torch.float32),
            "target_t": torch.as_tensor(arrays["target_t"][row], dtype=torch.long),
            "sequence_id": self.cache.sequence_ids[sequence_index],
        }
