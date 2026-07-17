from copy import deepcopy
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Tools.Utils import calculate_mouse_distance


# ============================================================
# Config / 配置
# Change these values when processing another npy file or using another distance limit.
# 如果要处理其他 npy 文件，或者想换距离阈值，直接修改下面两个常量。
# ============================================================
INPUT_NPY_PATH = Path(__file__).resolve().parent / "calms21_task1_train.npy"
DISTANCE_LIMIT = 330

# CalMS21 videos are recorded at 30 Hz (30 frames per second). This is used
# to turn a position difference between two consecutive frames into a velocity.
# CalMS21 视频的采集帧率是 30 Hz（每秒 30 帧）。用它可以把相邻两帧的位置差
# 转换成速度（单位时间的位移）。
FRAME_RATE_HZ = 30
FRAME_TIME = 1.0 / FRAME_RATE_HZ  # seconds per frame / 每一帧对应的秒数

# Mouse id convention used throughout CalMS21: 0 = resident (black mouse),
# 1 = intruder (white mouse). This matches the "mouse ID" axis of "keypoints".
# CalMS21 中鼠的编号约定：0 号是 resident（原住的黑色鼠），1 号是 intruder
# （入侵的白色鼠），和 "keypoints" 数组里 "mouse ID" 这一维的顺序一致。
RESIDENT_ID = 0
INTRUDER_ID = 1

# The 7 tracked body parts, in the exact order they appear along the last axis
# of the "keypoints" array. The original CalMS21 order is:
# (nose, left ear, right ear, neck, left hip, right hip, tail base).
# We keep that same order but use the short "node" names used in this script,
# so index 0 below always corresponds to "nose", index 3 to "neck", etc.
# 7 个追踪的身体部位，顺序和 "keypoints" 数组最后一维的顺序完全一致。
# CalMS21 原始顺序是：(鼻子, 左耳, 右耳, 颈部, 左髋, 右髋, 尾根)。
# 这里顺序不变，只是用脚本里的简短 node 名字表示，下面第 0 个对应"鼻子"，
# 第 3 个对应"颈部"，以此类推。
NODE_NAMES = ["head", "headLeft", "headRight", "neck", "bodyLeft", "bodyRight", "tail"]
NODE_INDEX = {name: index for index, name in enumerate(NODE_NAMES)}

# Node pairs used for "internode_distance". Each entry is
# (label, node_name_a, node_name_b). ORDER MUST BE PRESERVED EXACTLY, since
# internode_distance is stored as a plain list of 12 numbers in this order
# (not as a dict), so downstream code reads it positionally.
# 用于计算 "internode_distance" 的节点对，每一项是 (标签, 节点a, 节点b)。
# 必须严格保持下面这个顺序：internode_distance 存成一个长度为 12 的列表
# （不是字典），后续代码是按位置（下标）读取这些距离值的。
INTERNODE_PAIRS = [
    ("head2headLeft", "head", "headLeft"),
    ("head2headRight", "head", "headRight"),
    ("headRight2neck", "headRight", "neck"),
    ("headLeft2neck", "headLeft", "neck"),
    ("head2neck", "head", "neck"),
    ("neck2bodyLeft", "neck", "bodyLeft"),
    ("neck2bodyRight", "neck", "bodyRight"),
    ("neck2tail", "neck", "tail"),
    ("bodyLeft2tail", "bodyLeft", "tail"),
    ("bodyRight2tail", "bodyRight", "tail"),
    ("headL2headR", "headLeft", "headRight"),
    ("bodyL2bodyR", "bodyLeft", "bodyRight"),
]


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


# ============================================================
# Window + per-frame feature construction
#
# Final structure built below:
#
# Data[trial_id] = [window_1, window_2, ...]
# window_i        = [intruder_frames, resident_frames]
# intruder_frames  = [Int_Frame_1, Int_Frame_2, ...]
# resident_frames  = [Res_Frame_1, Res_Frame_2, ...]
#
# Each Int_Frame_i / Res_Frame_i is a dict:
# {
#   "body": {
#       "node": {
#           <node_name>: {
#               "position": (x, y),
#               "node_velocity": (vx, vy),
#               "score": float,   # CalMS21's own tracking-confidence score
#                                 # for this node, 0 (low) to 1 (high)
#           },
#           ... for all 7 nodes in NODE_NAMES order ...
#       },
#       "internode_distance": [12 floats, in INTERNODE_PAIRS order],
#   },
#   "interaction": {
#       "annotation_tag": one-hot list, length 4, ordered
#           [attack, investigation, mount, other]:
#           attack        -> [1, 0, 0, 0]
#           investigation -> [0, 1, 0, 0]
#           mount         -> [0, 0, 1, 0]
#           other         -> [0, 0, 0, 1]
#       "intruder_or_resident_tag": 0 (resident) or 1 (intruder),
#       "mouse_midpoint_distance": float,
#       "head_angle": float (signed degrees, -180..180),
#   }
# }
#
# 下面构建的最终结构：
# Data[trial_id] = [window_1, window_2, ...]
# window_i        = [intruder_frames, resident_frames]
# intruder_frames  = [Int_Frame_1, Int_Frame_2, ...]
# resident_frames  = [Res_Frame_1, Res_Frame_2, ...]
# 每个 Int_Frame_i / Res_Frame_i 都是上面这样的字典结构。
# 每个 node 除了 position / node_velocity，还新增了 "score"
# （CalMS21 原始数据自带的、该 node 的追踪置信度，0~1）。
# annotation_tag 是长度为 4 的 one-hot 列表，顺序固定为
# [attack, investigation, mount, other]，见上面英文注释中的对应关系。
# ============================================================


def get_contiguous_windows(keep_mask):
    """Split a boolean keep_mask into contiguous runs of True.

    把布尔数组 keep_mask 切分成若干段连续为 True 的片段。
    A "window" is exactly one such contiguous run: a stretch of the ORIGINAL,
    time-ordered sequence where the mouse-center distance stayed below
    DISTANCE_LIMIT on every frame. This is why velocities inside a window are
    always computed from two genuinely consecutive video frames.
    一个 "window" 就是这样一段连续区间：在原始（未被打乱顺序的）序列中，
    两鼠中心距离连续小于 DISTANCE_LIMIT 的一段。这保证了 window 内部计算
    速度时，前后两帧永远是真实相邻的视频帧。

    Returns a list of (start, end) tuples, end EXCLUSIVE, i.e. frame indices
    range(start, end) all belong to the same window.
    返回 (start, end) 元组列表，end 是开区间右端点，即 range(start, end)
    范围内的帧属于同一个 window。
    """
    windows = []
    n_frames = len(keep_mask)
    t = 0
    while t < n_frames:
        if keep_mask[t]:
            start = t
            while t < n_frames and keep_mask[t]:
                t += 1
            windows.append((start, t))
        else:
            t += 1
    return windows


def get_node_position(keypoints, frame_idx, mouse_id, node_name):
    """Return the (x, y) position of one node, for one mouse, at one frame.

    返回某一帧、某一只鼠、某一个 node 的 (x, y) 坐标。
    keypoints has shape (frames, mouse, xy, node).
    keypoints 的形状是 (帧数, 鼠, xy坐标, node)。
    """
    node_idx = NODE_INDEX[node_name]
    return keypoints[frame_idx, mouse_id, :, node_idx].astype(float)


def get_node_score(scores, frame_idx, mouse_id, node_name):
    """Return CalMS21's own tracking-confidence score for one node.

    返回 CalMS21 原始数据里，某一帧、某一只鼠、某一个 node 的追踪置信度分数。
    "scores" has shape (frames, mouse, node), unitless, range 0 (lowest
    confidence) to 1 (highest confidence) -- this is a value CalMS21 already
    provides per node, we are not computing it ourselves.
    "scores" 的形状是 (帧数, 鼠, node)，取值范围 0（置信度最低）到
    1（置信度最高），这是 CalMS21 数据自带的值，不是脚本计算出来的。
    """
    node_idx = NODE_INDEX[node_name]
    return float(scores[frame_idx, mouse_id, node_idx])


def compute_node_velocity(keypoints, frame_idx, window_start, mouse_id, node_name):
    """Compute one node's velocity as (pos(t) - pos(t-1)) / frame_time.

    按 (pos(t)-pos(t-1)) / frame_time 计算某个 node 的速度。
    For the FIRST frame of a window there is no in-window previous frame to
    diff against, so by convention we set the velocity to (0, 0) for that
    frame only.
    每个 window 的第一帧没有 window 内部的"上一帧"可用于计算速度，
    按约定这种情况下速度记为 (0, 0)，仅对该帧生效。
    Units: pixels per second (position is in pixels, FRAME_TIME is in seconds).
    单位：像素/秒（位置单位是像素，FRAME_TIME 单位是秒）。
    """
    if frame_idx == window_start:
        return np.zeros(2, dtype=float)

    current_pos = get_node_position(keypoints, frame_idx, mouse_id, node_name)
    previous_pos = get_node_position(keypoints, frame_idx - 1, mouse_id, node_name)
    return (current_pos - previous_pos) / FRAME_TIME


def compute_mouse_midpoint(keypoints, frame_idx, mouse_id):
    """Mean (x, y) position of all 7 nodes for one mouse at one frame.

    计算某一帧中，某只鼠 7 个 node 位置的平均值（即该鼠的中点/中心位置）。
    """
    all_node_positions = keypoints[frame_idx, mouse_id, :, :]  # shape (xy=2, node=7)
    return all_node_positions.mean(axis=1).astype(float)  # shape (2,)


def compute_internode_distances(keypoints, frame_idx, mouse_id):
    """Compute the 12 predefined internode distances, in INTERNODE_PAIRS order.

    按 INTERNODE_PAIRS 规定的顺序，计算 12 个预先定义好的节点间距离。
    Returns a plain list of 12 floats (order preserved, see INTERNODE_PAIRS).
    返回长度为 12 的浮点数列表（顺序见 INTERNODE_PAIRS，务必保持不变）。
    """
    distances = []
    for _label, node_a, node_b in INTERNODE_PAIRS:
        pos_a = get_node_position(keypoints, frame_idx, mouse_id, node_a)
        pos_b = get_node_position(keypoints, frame_idx, mouse_id, node_b)
        distances.append(float(np.linalg.norm(pos_a - pos_b)))
    return distances


def compute_head_angle(keypoints, frame_idx, mouse_id, other_mouse_midpoint):
    """Signed angle (degrees, range (-180, 180]) between:
      - this mouse's neck->nose heading vector, and
      - this mouse's nose->other_mouse_midpoint vector.

    计算带符号夹角（单位：度，范围 (-180, 180]），介于：
      - 该鼠自身 neck->nose（颈部到鼻子）的朝向向量；
      - 该鼠 nose 指向对方鼠中点（mouse_midpoint）的向量。
    A positive value means the other mouse is to the counter-clockwise side of
    where this mouse is facing; negative means the clockwise side. This uses
    the standard atan2(cross, dot) formula for a signed 2D angle.
    正值表示对方鼠位于该鼠朝向的逆时针一侧，负值表示顺时针一侧。
    使用标准的 atan2(cross, dot) 公式计算带符号的二维夹角。
    """
    nose_pos = get_node_position(keypoints, frame_idx, mouse_id, "head")
    neck_pos = get_node_position(keypoints, frame_idx, mouse_id, "neck")

    heading_vector = nose_pos - neck_pos
    to_other_vector = other_mouse_midpoint - nose_pos

    cross = (
        heading_vector[0] * to_other_vector[1]
        - heading_vector[1] * to_other_vector[0]
    )
    dot = (
        heading_vector[0] * to_other_vector[0]
        + heading_vector[1] * to_other_vector[1]
    )

    angle_radians = np.arctan2(cross, dot)
    return float(np.degrees(angle_radians))


def build_body_section(keypoints, scores, frame_idx, window_start, mouse_id):
    """Build the "body" sub-dictionary for one mouse at one frame.

    为某只鼠在某一帧构建 "body" 字段：包含每个 node 的位置/速度/置信度分数，
    以及 12 个 internode_distance。
    """
    node_section = {}
    for node_name in NODE_NAMES:
        position = get_node_position(keypoints, frame_idx, mouse_id, node_name)
        velocity = compute_node_velocity(
            keypoints, frame_idx, window_start, mouse_id, node_name
        )
        score = get_node_score(scores, frame_idx, mouse_id, node_name)
        node_section[node_name] = {
            "position": (float(position[0]), float(position[1])),
            "node_velocity": (float(velocity[0]), float(velocity[1])),
            "score": score,
        }

    return {
        "node": node_section,
        "internode_distance": compute_internode_distances(keypoints, frame_idx, mouse_id),
    }


def build_interaction_section(
    keypoints, frame_idx, mouse_id, other_mouse_id, annotation_onehot
):
    """Build the "interaction" sub-dictionary for one mouse at one frame.

    为某只鼠在某一帧构建 "interaction" 字段：annotation_tag、
    intruder_or_resident_tag、mouse_midpoint_distance、head_angle。
    """
    own_midpoint = compute_mouse_midpoint(keypoints, frame_idx, mouse_id)
    other_midpoint = compute_mouse_midpoint(keypoints, frame_idx, other_mouse_id)
    mouse_midpoint_distance = float(np.linalg.norm(own_midpoint - other_midpoint))
    head_angle = compute_head_angle(keypoints, frame_idx, mouse_id, other_midpoint)

    return {
        "annotation_tag": annotation_onehot,
        "intruder_or_resident_tag": 0 if mouse_id == RESIDENT_ID else 1,
        "mouse_midpoint_distance": mouse_midpoint_distance,
        "head_angle": head_angle,
    }


def build_frame(
    keypoints,
    scores,
    frame_idx,
    window_start,
    mouse_id,
    other_mouse_id,
    annotation_onehot,
):
    """Build one frame dict (an "Int_Frame_i" or "Res_Frame_i") for one mouse.

    构建单只鼠、单帧的数据字典（对应 "Int_Frame_i" 或 "Res_Frame_i"）。
    """
    return {
        "body": build_body_section(keypoints, scores, frame_idx, window_start, mouse_id),
        "interaction": build_interaction_section(
            keypoints, frame_idx, mouse_id, other_mouse_id, annotation_onehot
        ),
    }


def get_vocab_order(sequence_data):
    """Recover behavior names ordered by their integer id.

    从序列自身的 metadata 中，按行为 id 从小到大还原出行为名称列表。
    metadata['vocab'] maps behavior name -> integer id, e.g.
    {'attack': 0, 'investigation': 1, 'mount': 2, 'other': 3}, and is the
    same for every sequence in the dataset (per CalMS21 documentation).
    metadata['vocab'] 把行为名称映射到整数 id，例如：
    {'attack': 0, 'investigation': 1, 'mount': 2, 'other': 3}，
    根据 CalMS21 官方说明，整个数据集里所有序列共用同一份 vocab。
    """
    vocab = sequence_data["metadata"]["vocab"]
    return [name for name, _id in sorted(vocab.items(), key=lambda item: item[1])]


def build_annotation_onehot(annotation_id, vocab_order):
    """One-hot-encode a frame's annotation id, ordered by vocab_order (id order).

    把某一帧的标注 id 转换成 one-hot 列表，顺序按 vocab_order（即行为 id 顺序）。
    With CalMS21's vocab {'attack': 0, 'investigation': 1, 'mount': 2,
    'other': 3}, vocab_order sorted by id is
    ["attack", "investigation", "mount", "other"], which gives exactly:
        attack        (id 0) -> [1, 0, 0, 0]
        investigation (id 1) -> [0, 1, 0, 0]
        mount         (id 2) -> [0, 0, 1, 0]
        other         (id 3) -> [0, 0, 0, 1]
    对于 CalMS21 的 vocab {'attack': 0, 'investigation': 1, 'mount': 2,
    'other': 3}，按 id 排序后的 vocab_order 正好是
    ["attack", "investigation", "mount", "other"]，因此：
        attack        (id 0) -> [1, 0, 0, 0]
        investigation (id 1) -> [0, 1, 0, 0]
        mount         (id 2) -> [0, 0, 1, 0]
        other         (id 3) -> [0, 0, 0, 1]
    """
    onehot = [0] * len(vocab_order)
    onehot[annotation_id] = 1
    return onehot


def build_window(sequence_data, window_start, window_end, vocab_order):
    """Build one window: [intruder_frames, resident_frames].

    构建一个 window：[intruder_frames, resident_frames]。
    intruder_frames / resident_frames are lists of frame dicts
    (Int_Frame_i / Res_Frame_i), one per frame in [window_start, window_end).
    intruder_frames / resident_frames 分别是帧字典（Int_Frame_i / Res_Frame_i）
    组成的列表，对应 [window_start, window_end) 范围内的每一帧。
    """
    keypoints = sequence_data["keypoints"]
    scores = sequence_data["scores"]
    annotations = sequence_data["annotations"]

    intruder_frames = []
    resident_frames = []

    for frame_idx in range(window_start, window_end):
        annotation_onehot = build_annotation_onehot(
            int(annotations[frame_idx]), vocab_order
        )

        resident_frame = build_frame(
            keypoints,
            scores,
            frame_idx,
            window_start,
            mouse_id=RESIDENT_ID,
            other_mouse_id=INTRUDER_ID,
            annotation_onehot=annotation_onehot,
        )
        intruder_frame = build_frame(
            keypoints,
            scores,
            frame_idx,
            window_start,
            mouse_id=INTRUDER_ID,
            other_mouse_id=RESIDENT_ID,
            annotation_onehot=annotation_onehot,
        )

        intruder_frames.append(intruder_frame)
        resident_frames.append(resident_frame)

    return [intruder_frames, resident_frames]


def build_trial_windows(sequence_data, distance_limit):
    """Break one trial (sequence) into windows by mouse-center distance, and
    build the full nested frame-feature structure for every window.

    利用两鼠中心距离，把一个 trial（实验序列）切分成若干个 window，
    并为每个 window 构建完整的嵌套帧特征结构。
    Returns a list of windows: [window_1, window_2, ...].
    返回一个 window 列表：[window_1, window_2, ...]。
    """
    distances = calculate_mouse_distance(sequence_data["keypoints"])
    keep_mask = distances < distance_limit

    vocab_order = get_vocab_order(sequence_data)

    windows = []
    for window_start, window_end in get_contiguous_windows(keep_mask):
        windows.append(build_window(sequence_data, window_start, window_end, vocab_order))

    return windows


def build_windowed_dataset(dataset, distance_limit):
    """Build the final windowed, per-frame-feature dataset for every trial.

    为数据集中的每一个 trial 构建最终的、按 window 组织的逐帧特征数据。
    Returned structure (see the big comment block above build_windowed_dataset
    for the full nested layout):
    返回结构（完整的嵌套结构说明见上面的大段注释）：

    Data[trial_id] = [window_1, window_2, ...]

    NOTE: this assumes a single annotator/group per npy file, which is the
    case for CalMS21 Task 1 (the file this script is set up to process). If a
    dataset ever contains multiple groups, trial_id keys from different groups
    would collide; that situation is not handled here.
    注意：这里假设每个 npy 文件只有一个 annotator/group，这对于本脚本处理的
    CalMS21 Task 1 数据是成立的。如果数据集包含多个 group，不同 group 下
    相同的 trial_id 会互相覆盖，这里没有处理这种情况。
    """
    data = {}
    for group_name, sequences in dataset.items():
        for sequence_id, sequence_data in sequences.items():
            windows = build_trial_windows(sequence_data, distance_limit)
            total_frames = sum(len(window[0]) for window in windows)
            data[sequence_id] = windows
            print(
                f"{sequence_id}: {len(windows)} windows built "
                f"({total_frames} total frames kept)"
            )

    return data


def build_windowed_output_path(input_path, distance_limit):
    """Build an output filename for the windowed/feature dataset.

    为窗口化 + 特征化后的数据集生成输出文件名。
    """
    distance_label = str(distance_limit).replace(".", "p")
    return input_path.with_name(
        f"{input_path.stem}_windowed_distance_lt_{distance_label}.npy"
    )


def main():
    """Load the npy file, then build BOTH outputs:
      1) the original flat, distance-filtered dataset (unchanged behavior);
      2) the new windowed, per-frame-feature dataset described above.

    加载原始 npy 文件，然后同时生成两种输出：
      1）原有的、按距离筛选后的扁平数据集（行为不变）；
      2）新增的、按 window 组织并带逐帧特征的数据集。
    """
    input_path = INPUT_NPY_PATH
    filtered_output_path = build_output_path(input_path, DISTANCE_LIMIT)
    windowed_output_path = build_windowed_output_path(input_path, DISTANCE_LIMIT)

    print(f"Input npy: {input_path}")
    print(f"Distance limit: mouse center distance < {DISTANCE_LIMIT}")
    print(f"Filtered output npy: {filtered_output_path}")
    print(f"Windowed output npy: {windowed_output_path}")

    dataset = np.load(input_path, allow_pickle=True).item()

    # --- 1) Original flat, distance-filtered dataset ---
    # --- 1）原有的、按距离筛选后的扁平数据集 ---
    filtered_dataset, total_kept, total_frames = filter_dataset_by_distance(
        dataset, DISTANCE_LIMIT
    )
    np.save(filtered_output_path, filtered_dataset, allow_pickle=True)

    keep_ratio = total_kept / total_frames if total_frames else 0
    print("---------------------------------")
    print(f"Saved: {filtered_output_path}")
    print(f"Total: {total_frames} -> {total_kept} frames kept")
    print(f"Keep ratio: {keep_ratio:.2%}")

    # --- 2) New windowed, per-frame-feature dataset ---
    # --- 2）新增的、按 window 组织并带逐帧特征的数据集 ---
    print("---------------------------------")
    print("Building windowed per-frame feature dataset...")
    windowed_data = build_windowed_dataset(dataset, DISTANCE_LIMIT)
    np.save(windowed_output_path, windowed_data, allow_pickle=True)
    print(f"Saved: {windowed_output_path}")


if __name__ == "__main__":
    main()
