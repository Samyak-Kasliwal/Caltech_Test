import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

if __package__ is None or __package__ == "":
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from Model.asymmetric_pose_case.data import (
        NOSE_INDEX,
        NECK_INDEX,
        keypoints_to_joint_xy,
        load_calms21_sequences,
    )
    from Model.asymmetric_pose_case.runtime import (
        build_modules_from_checkpoint,
        forward_batch,
        resolve_device,
        to_jsonable,
    )
    from Model.asymmetric_pose_case.validation.config import RolloutConfig
    from Model.asymmetric_pose_case.validation.metrics import compute_pose_errors
else:
    from ..data import NOSE_INDEX, NECK_INDEX, keypoints_to_joint_xy, load_calms21_sequences
    from ..runtime import (
        build_modules_from_checkpoint,
        forward_batch,
        resolve_device,
        to_jsonable,
    )
    from .config import RolloutConfig
    from .metrics import compute_pose_errors


def save_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def find_record(records, sequence_id: str):
    for record in records:
        if record.sequence_id == sequence_id:
            return record
    available = "\n".join(record.sequence_id for record in records[:10])
    raise ValueError(
        f"Unknown sequence_id: {sequence_id}\nFirst available sequence ids:\n{available}"
    )


def compute_joint_velocity_xy(joint_xy: np.ndarray) -> np.ndarray:
    velocity = np.zeros_like(joint_xy, dtype=np.float32)
    velocity[1:] = joint_xy[1:] - joint_xy[:-1]
    return velocity


def compute_pair_interaction_xy(a_xy: np.ndarray, b_xy: np.ndarray) -> np.ndarray:
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
    return np.concatenate(
        [center_delta, center_distance, angle_sin, angle_cos], axis=1
    ).astype(np.float32)


def labels_for_indices(annotations: np.ndarray, frame_indices: np.ndarray, mode: str) -> np.ndarray:
    if mode == "history_plus_current":
        return annotations[frame_indices]
    if mode == "history_only":
        return annotations[np.maximum(frame_indices - 1, 0)]
    if mode == "none":
        return np.zeros(len(frame_indices), dtype=np.int64)
    raise ValueError(f"Unknown behavior_label_mode: {mode}")


def normalize_tensor(value: np.ndarray, normalizer, device: torch.device) -> torch.Tensor:
    tensor = torch.tensor(value, dtype=torch.float32, device=device)
    return normalizer.transform(tensor).unsqueeze(0)


def build_rollout_batch(
    cfg,
    normalizers,
    a_true_xy: np.ndarray,
    b_buffer_xy: np.ndarray,
    annotations: np.ndarray,
    target_t: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    a_indices = np.arange(target_t - cfg.data.a_history_frames, target_t + 1)
    b_indices = np.arange(target_t - cfg.data.b_history_frames, target_t)

    # Build asymmetric model inputs from true A and rollout B history.
    # 用真实 A 和递归预测得到的 B 历史构建非对称模型输入。
    a_xy = a_true_xy[a_indices]
    b_xy = b_buffer_xy[b_indices]
    a_velocity = compute_joint_velocity_xy(a_true_xy)[a_indices]
    b_velocity = compute_joint_velocity_xy(b_buffer_xy)[b_indices]
    interaction = compute_pair_interaction_xy(a_true_xy[b_indices], b_buffer_xy[b_indices])

    batch = {
        "a_xy": normalize_tensor(a_xy, normalizers.coord, device),
        "b_xy": normalize_tensor(b_xy, normalizers.coord, device),
        "a_velocity": normalize_tensor(a_velocity, normalizers.velocity, device),
        "b_velocity": normalize_tensor(b_velocity, normalizers.velocity, device),
        "interaction": normalize_tensor(interaction, normalizers.interaction, device),
        "a_behavior": torch.tensor(
            labels_for_indices(annotations, a_indices, cfg.data.behavior_label_mode),
            dtype=torch.long,
            device=device,
        ).unsqueeze(0),
        "b_behavior": torch.tensor(
            labels_for_indices(annotations, b_indices, cfg.data.behavior_label_mode),
            dtype=torch.long,
            device=device,
        ).unsqueeze(0),
        "current_behavior": torch.tensor([annotations[target_t]], dtype=torch.long, device=device),
    }
    return batch


@torch.no_grad()
def run_rollout(rollout_cfg: RolloutConfig) -> Path:
    if not rollout_cfg.use_true_a_future:
        raise ValueError("This rollout implementation currently requires true future A frames.")
    if not rollout_cfg.use_true_annotation_future:
        raise ValueError("This rollout implementation currently requires true future annotation labels.")

    device = resolve_device("auto")

    # Load checkpoint, rebuild model modules, and restore training normalizers.
    # 加载 checkpoint，重建模型模块，并恢复训练时保存的归一化参数。
    modules, normalizers, checkpoint = build_modules_from_checkpoint(
        rollout_cfg.checkpoint_path, device
    )
    cfg = checkpoint["config"]
    from Model.asymmetric_pose_case.runtime import config_from_dict

    cfg = config_from_dict(cfg)
    for module in modules.values():
        module.eval()

    records = load_calms21_sequences(cfg.data.data_path)
    record = find_record(records, rollout_cfg.sequence_id)
    a_true_xy = keypoints_to_joint_xy(record.keypoints, 0)
    b_true_xy = keypoints_to_joint_xy(record.keypoints, 1)
    annotations = record.annotations

    start_t = rollout_cfg.start_t
    end_t = start_t + rollout_cfg.rollout_length
    if start_t < cfg.data.min_target_index:
        raise ValueError(f"start_t must be >= {cfg.data.min_target_index}.")
    if end_t > len(annotations):
        raise ValueError(
            f"Rollout end {end_t} exceeds sequence length {len(annotations)}."
        )

    b_rollout_buffer = b_true_xy.copy()
    a_pose_xy: list[np.ndarray] = []
    b_pred_pose_xy: list[np.ndarray] = []
    b_true_pose_xy: list[np.ndarray] = []
    b_pred_delta_xy: list[np.ndarray] = []
    b_true_delta_xy: list[np.ndarray] = []
    target_t_values: list[int] = []

    for target_t in range(start_t, end_t):
        batch = build_rollout_batch(
            cfg,
            normalizers,
            a_true_xy,
            b_rollout_buffer,
            annotations,
            target_t,
            device,
        )

        prediction_norm = forward_batch(batch, modules, cfg, device)

        # Convert model delta output back to absolute joint coordinates.
        # 将模型输出的位移还原成绝对关节坐标，方便直接接可视化函数。
        b_pred_delta = normalizers.target_delta.inverse(prediction_norm).cpu().numpy()[0]
        b_pred_delta = b_pred_delta.reshape(cfg.data.num_joints, cfg.data.coord_dim)
        b_previous_pose = b_rollout_buffer[target_t - 1]
        b_pred_pose = b_previous_pose + b_pred_delta

        # Reuse predicted B poses for future B-history inputs.
        # 将已经预测出的 B 姿态写回历史窗口，供后续帧继续递归预测。
        b_rollout_buffer[target_t] = b_pred_pose

        a_pose_xy.append(a_true_xy[target_t])
        b_pred_pose_xy.append(b_pred_pose)
        b_true_pose_xy.append(b_true_xy[target_t])
        b_pred_delta_xy.append(b_pred_delta)
        b_true_delta_xy.append(b_true_xy[target_t] - b_true_xy[target_t - 1])
        target_t_values.append(target_t)

    a_pose_xy_arr = np.asarray(a_pose_xy, dtype=np.float32)
    b_pred_pose_xy_arr = np.asarray(b_pred_pose_xy, dtype=np.float32)
    b_true_pose_xy_arr = np.asarray(b_true_pose_xy, dtype=np.float32)
    b_pred_delta_xy_arr = np.asarray(b_pred_delta_xy, dtype=np.float32)
    b_true_delta_xy_arr = np.asarray(b_true_delta_xy, dtype=np.float32)
    target_t_arr = np.asarray(target_t_values, dtype=np.int64)
    errors = compute_pose_errors(b_pred_pose_xy_arr, b_true_pose_xy_arr)

    keypoints_pred_pair = np.empty(
        (rollout_cfg.rollout_length, 2, 2, cfg.data.num_joints), dtype=np.float32
    )
    keypoints_pred_pair[:, 0] = np.transpose(a_pose_xy_arr, (0, 2, 1))
    keypoints_pred_pair[:, 1] = np.transpose(b_pred_pose_xy_arr, (0, 2, 1))

    rollout_cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Save visualization-friendly absolute coordinates and per-frame errors.
    # 保存适合可视化的绝对坐标，以及每一帧的误差信息。
    np.savez_compressed(
        rollout_cfg.output_path,
        a_pose_xy=a_pose_xy_arr,
        b_pred_pose_xy=b_pred_pose_xy_arr,
        b_true_pose_xy=b_true_pose_xy_arr,
        b_pred_delta_xy=b_pred_delta_xy_arr,
        b_true_delta_xy=b_true_delta_xy_arr,
        per_frame_mse=errors["per_frame_mse"],
        per_frame_rmse=errors["per_frame_rmse"],
        per_joint_l2=errors["per_joint_l2"],
        target_t=target_t_arr,
        sequence_id=np.asarray(rollout_cfg.sequence_id),
        start_t=np.asarray(start_t),
        rollout_length=np.asarray(rollout_cfg.rollout_length),
        keypoints_pred_pair=keypoints_pred_pair,
    )

    summary = {
        "sequence_id": rollout_cfg.sequence_id,
        "start_t": start_t,
        "rollout_length": rollout_cfg.rollout_length,
        "checkpoint_path": str(rollout_cfg.checkpoint_path),
        "output_path": str(rollout_cfg.output_path),
        "mean_mse": errors["mean_mse"],
        "mean_rmse": errors["mean_rmse"],
        "mean_joint_l2": errors["mean_joint_l2"],
        "final_frame_mse": errors["final_frame_mse"],
        "final_frame_rmse": errors["final_frame_rmse"],
    }
    save_json(rollout_cfg.summary_path, summary)
    print(f"Saved rollout predictions: {rollout_cfg.output_path}")
    print(f"Saved rollout summary: {rollout_cfg.summary_path}")
    return rollout_cfg.output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--checkpoint-name", type=str, default="best.pt")
    parser.add_argument("--sequence-id", type=str, required=True)
    parser.add_argument("--start-t", type=int, required=True)
    parser.add_argument("--rollout-length", type=int, default=20)
    parser.add_argument("--output-name", type=str, default="rollout_predictions.npz")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_rollout(
        RolloutConfig(
            run_dir=args.run_dir,
            fold=args.fold,
            checkpoint_name=args.checkpoint_name,
            sequence_id=args.sequence_id,
            start_t=args.start_t,
            rollout_length=args.rollout_length,
            output_name=args.output_name,
        )
    )
