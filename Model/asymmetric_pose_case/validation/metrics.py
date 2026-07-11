import numpy as np


def compute_pose_errors(
    b_pred_pose_xy: np.ndarray, b_true_pose_xy: np.ndarray
) -> dict[str, np.ndarray | float]:
    diff = b_pred_pose_xy - b_true_pose_xy
    per_joint_l2 = np.linalg.norm(diff, axis=-1)
    per_frame_mse = np.mean(np.square(diff), axis=(1, 2))
    per_frame_rmse = np.sqrt(per_frame_mse)
    return {
        "per_joint_l2": per_joint_l2.astype(np.float32),
        "per_frame_mse": per_frame_mse.astype(np.float32),
        "per_frame_rmse": per_frame_rmse.astype(np.float32),
        "mean_mse": float(np.mean(per_frame_mse)),
        "mean_rmse": float(np.mean(per_frame_rmse)),
        "mean_joint_l2": float(np.mean(per_joint_l2)),
        "final_frame_mse": float(per_frame_mse[-1]),
        "final_frame_rmse": float(per_frame_rmse[-1]),
    }
