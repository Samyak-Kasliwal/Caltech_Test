from dataclasses import dataclass
import random

from .data import SequenceRecord, WindowIndex


@dataclass
class FoldSplit:
    fold_index: int
    train_windows: list[WindowIndex]
    val_windows: list[WindowIndex]
    train_sequence_ids: set[str]
    val_sequence_ids: set[str]


def build_sequence_kfold_splits(
    records: list[SequenceRecord],
    windows: list[WindowIndex],
    num_folds: int,
    seed: int,
    holdout_val_ratio: float = 0.2,
) -> list[FoldSplit]:
    sequence_indices = list(range(len(records)))
    rng = random.Random(seed)
    rng.shuffle(sequence_indices)

    if num_folds == 1:
        val_count = max(1, round(len(sequence_indices) * holdout_val_ratio))
        val_sequence_indices = sequence_indices[:val_count]
        val_sequence_set = set(val_sequence_indices)
        train_windows = [
            window for window in windows if window.sequence_index not in val_sequence_set
        ]
        val_windows = [
            window for window in windows if window.sequence_index in val_sequence_set
        ]
        return [
            FoldSplit(
                fold_index=0,
                train_windows=train_windows,
                val_windows=val_windows,
                train_sequence_ids={
                    records[index].sequence_id
                    for index in sequence_indices
                    if index not in val_sequence_set
                },
                val_sequence_ids={
                    records[index].sequence_id for index in val_sequence_indices
                },
            )
        ]

    folds = [sequence_indices[i::num_folds] for i in range(num_folds)]
    splits: list[FoldSplit] = []
    for fold_index, val_sequence_indices in enumerate(folds):
        val_sequence_set = set(val_sequence_indices)
        train_windows = [
            window for window in windows if window.sequence_index not in val_sequence_set
        ]
        val_windows = [
            window for window in windows if window.sequence_index in val_sequence_set
        ]
        train_sequence_ids = {
            records[index].sequence_id
            for index in sequence_indices
            if index not in val_sequence_set
        }
        val_sequence_ids = {
            records[index].sequence_id for index in val_sequence_indices
        }
        splits.append(
            FoldSplit(
                fold_index=fold_index,
                train_windows=train_windows,
                val_windows=val_windows,
                train_sequence_ids=train_sequence_ids,
                val_sequence_ids=val_sequence_ids,
            )
        )
    return splits


def build_window_random_splits(
    records: list[SequenceRecord],
    windows: list[WindowIndex],
    num_folds: int,
    seed: int,
    holdout_val_ratio: float = 0.2,
) -> list[FoldSplit]:
    shuffled_windows = list(windows)
    rng = random.Random(seed)
    rng.shuffle(shuffled_windows)

    if num_folds == 1:
        val_count = max(1, round(len(shuffled_windows) * holdout_val_ratio))
        val_windows = shuffled_windows[:val_count]
        train_windows = shuffled_windows[val_count:]
        return [
            FoldSplit(
                fold_index=0,
                train_windows=train_windows,
                val_windows=val_windows,
                train_sequence_ids={
                    records[window.sequence_index].sequence_id for window in train_windows
                },
                val_sequence_ids={
                    records[window.sequence_index].sequence_id for window in val_windows
                },
            )
        ]

    fold_windows = [shuffled_windows[i::num_folds] for i in range(num_folds)]

    splits: list[FoldSplit] = []
    for fold_index, val_windows in enumerate(fold_windows):
        val_set = {(window.sequence_index, window.target_t) for window in val_windows}
        train_windows = [
            window
            for window in shuffled_windows
            if (window.sequence_index, window.target_t) not in val_set
        ]
        splits.append(
            FoldSplit(
                fold_index=fold_index,
                train_windows=train_windows,
                val_windows=val_windows,
                train_sequence_ids={
                    records[window.sequence_index].sequence_id for window in train_windows
                },
                val_sequence_ids={
                    records[window.sequence_index].sequence_id for window in val_windows
                },
            )
        )
    return splits


def build_kfold_splits(
    records: list[SequenceRecord],
    windows: list[WindowIndex],
    split_mode: str,
    num_folds: int,
    seed: int,
    holdout_val_ratio: float = 0.2,
) -> list[FoldSplit]:
    if num_folds < 1:
        raise ValueError("num_folds must be >= 1.")
    if not 0 < holdout_val_ratio < 1:
        raise ValueError("holdout_val_ratio must be between 0 and 1.")

    if split_mode == "sequence_level":
        return build_sequence_kfold_splits(
            records, windows, num_folds, seed, holdout_val_ratio
        )

    # To switch to random window-level folds, set TrainingConfig.split_mode = "window_level".
    # 如需改为 window 随机划分，将 TrainingConfig.split_mode 设置为 "window_level"。
    if split_mode == "window_level":
        return build_window_random_splits(
            records, windows, num_folds, seed, holdout_val_ratio
        )

    raise ValueError(f"Unsupported split_mode: {split_mode}")
