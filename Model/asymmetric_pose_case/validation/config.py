from dataclasses import dataclass
from pathlib import Path


@dataclass
class RolloutConfig:
    # Directory produced by train.py, containing config.json and fold folders.
    # train.py 生成的结果目录，里面包含 config.json 和各个 fold 子目录。
    run_dir: Path

    # One-based fold index; fold=1 maps to fold_01.
    # 从 1 开始的 fold 编号；fold=1 对应 fold_01。
    fold: int = 1

    # Checkpoint filename under fold_XX/checkpoints.
    # fold_XX/checkpoints 下面的 checkpoint 文件名。
    checkpoint_name: str = "best.pt"

    # CalMS21 sequence id used as the rollout source.
    # 作为 rollout 起点来源的 CalMS21 sequence id。
    sequence_id: str = ""

    # First target frame to predict.
    # 第一个需要预测的目标帧。
    start_t: int = 100

    # Number of consecutive frames to predict; 20 is only the default.
    # 连续预测帧数；20 只是默认值，可以随时调整。
    rollout_length: int = 20

    # Output filename inside fold_XX/rollouts.
    # fold_XX/rollouts 目录下的输出文件名。
    output_name: str = "rollout_predictions.npz"

    # Use true future A frames as known conditioning inputs.
    # 使用真实未来 A 帧作为已知条件输入。
    use_true_a_future: bool = True

    # Use true future global annotation labels as conditioning inputs.
    # 使用真实未来全局 annotation 标签作为条件输入。
    use_true_annotation_future: bool = True

    @property
    def fold_dir(self) -> Path:
        return self.run_dir / f"fold_{self.fold:02d}"

    @property
    def checkpoint_path(self) -> Path:
        return self.fold_dir / "checkpoints" / self.checkpoint_name

    @property
    def output_dir(self) -> Path:
        return self.fold_dir / "rollouts"

    @property
    def output_path(self) -> Path:
        return self.output_dir / self.output_name

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "rollout_summary.json"
