import argparse
from pathlib import Path


def run_rollout(*_args, **_kwargs) -> Path:
    raise NotImplementedError(
        "Rollout validation must be rebuilt for the windowed feature dataset. "
        "The old rollout script used raw keypoints, while training now reads "
        "Caltech/calms21_task1_train_windowed_distance_lt_330.npy."
    )


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
    parse_args()
    run_rollout()
