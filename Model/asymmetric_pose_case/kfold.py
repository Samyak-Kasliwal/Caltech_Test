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
) -> list[FoldSplit]:
    sequence_indices = list(range(len(records)))
    rng = random.Random(seed)
    rng.shuffle(sequence_indices)

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
) -> list[FoldSplit]:
    shuffled_windows = list(windows)
    rng = random.Random(seed)
    rng.shuffle(shuffled_windows)
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
) -> list[FoldSplit]:
    if split_mode == "sequence_level":
        return build_sequence_kfold_splits(records, windows, num_folds, seed)

    # To switch to random window-level folds, set TrainingConfig.split_mode = "window_level".
    # 如需改为 window 随机划分，将 TrainingConfig.split_mode 设置为 "window_level"。
    if split_mode == "window_level":
        return build_window_random_splits(records, windows, num_folds, seed)

    raise ValueError(f"Unsupported split_mode: {split_mode}")
