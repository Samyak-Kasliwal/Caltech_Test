from copy import deepcopy
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Tools.Utils import calculate_mouse_distance


# Config / 配置:
# Change these values when processing another npy file or using another distance limit.
# 如果要处理其他 npy 文件，或者想换距离阈值，直接修改下面两个常量。
INPUT_NPY_PATH = Path(__file__).resolve().parent / "calms21_task1_train.npy"
DISTANCE_LIMIT = 330


def build_output_path(input_path, distance_limit):
    """Build an output filename that records the source file and distance limit.

    生成输出文件名，并在文件名里记录原始 npy 名称和距离限制。
    """
    distance_label = str(distance_limit).replace(".", "p")
    return input_path.with_name(
        f"{input_path.stem}_distance_lt_{distance_label}.npy"
    )


def filter_sequence_by_distance(sequence_data, distance_limit):
    """Filter one experiment sequence by mouse center distance.

    对单个实验序列进行筛选：只保留两只小鼠中心距离小于阈值的帧。
    """
    # Calculate the center-to-center distance for every frame.
    # 计算每一帧中两只小鼠中心点之间的距离。
    distances = calculate_mouse_distance(sequence_data["keypoints"])
    keep_mask = distances < distance_limit
    frame_count = len(distances)

    filtered_sequence = {}
    for key, value in sequence_data.items():
        # Frame-aligned arrays are filtered on the first dimension.
        # 与帧一一对应的数组按第一维进行筛选，例如 keypoints/scores/annotations。
        if isinstance(value, np.ndarray) and value.shape[:1] == (frame_count,):
            filtered_sequence[key] = value[keep_mask]
        else:
            # Non-frame data, such as metadata, is kept unchanged.
            # 非逐帧数据，例如 metadata，保持原样。
            filtered_sequence[key] = deepcopy(value)

    return filtered_sequence, int(np.sum(keep_mask)), frame_count


def filter_dataset_by_distance(dataset, distance_limit):
    """Filter all groups and experiment sequences in a CalMS21 npy dataset.

    遍历 CalMS21 npy 数据中的所有 annotator/group 和实验序列，并逐个筛选。
    """
    filtered_dataset = {}
    total_frames = 0
    total_kept = 0

    for group_name, sequences in dataset.items():
        filtered_dataset[group_name] = {}

        for sequence_id, sequence_data in sequences.items():
            filtered_sequence, kept_frames, frame_count = filter_sequence_by_distance(
                sequence_data, distance_limit
            )
            filtered_dataset[group_name][sequence_id] = filtered_sequence

            total_frames += frame_count
            total_kept += kept_frames

            print(f"{sequence_id}: {frame_count} -> {kept_frames} frames kept")
            if kept_frames == 0:
                # Keep the sequence with empty arrays, but warn the user.
                # 即使没有任何帧满足条件，也保留该序列为空数组，并打印提醒。
                print(
                    "WARNING: no frame satisfies "
                    f"distance < {distance_limit} in sequence {sequence_id}"
                )

    return filtered_dataset, total_kept, total_frames


def main():
    """Load the npy file, filter it, and save a new npy file.

    加载原始 npy，按距离筛选所有序列，然后保存为新的 npy 文件。
    """
    input_path = INPUT_NPY_PATH
    output_path = build_output_path(input_path, DISTANCE_LIMIT)

    print(f"Input npy: {input_path}")
    print(f"Distance limit: mouse center distance < {DISTANCE_LIMIT}")
    print(f"Output npy: {output_path}")

    dataset = np.load(input_path, allow_pickle=True).item()
    filtered_dataset, total_kept, total_frames = filter_dataset_by_distance(
        dataset, DISTANCE_LIMIT
    )

    # Save with the same nested dictionary format as the original file.
    # 按原始文件相同的嵌套字典格式保存。
    np.save(output_path, filtered_dataset, allow_pickle=True)

    keep_ratio = total_kept / total_frames if total_frames else 0
    print("---------------------------------")
    print(f"Saved: {output_path}")
    print(f"Total: {total_frames} -> {total_kept} frames kept")
    print(f"Keep ratio: {keep_ratio:.2%}")


if __name__ == "__main__":
    main()
