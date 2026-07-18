import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from Model.asymmetric_pose_case.config import default_config
    from Model.asymmetric_pose_case.data import (
        CalMS21AsymmetricPoseDataset,
        CachedCalMS21AsymmetricPoseDataset,
        build_or_load_window_feature_cache,
        build_window_indices,
        fit_normalizers,
        fit_normalizers_from_cache,
        load_calms21_sequences,
    )
    from Model.asymmetric_pose_case.kfold import build_kfold_splits
    from Model.asymmetric_pose_case.losses import build_loss
    from Model.asymmetric_pose_case.runtime import (
        build_modules,
        forward_batch,
        move_batch_to_device,
        print_preflight,
        resolve_device,
        save_checkpoint,
        to_jsonable,
    )
else:
    from .config import default_config
    from .data import (
        CalMS21AsymmetricPoseDataset,
        CachedCalMS21AsymmetricPoseDataset,
        build_or_load_window_feature_cache,
        build_window_indices,
        fit_normalizers,
        fit_normalizers_from_cache,
        load_calms21_sequences,
    )
    from .kfold import build_kfold_splits
    from .losses import build_loss
    from .runtime import (
        build_modules,
        forward_batch,
        move_batch_to_device,
        print_preflight,
        resolve_device,
        save_checkpoint,
        to_jsonable,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_run_dir(cfg) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{cfg.training.experiment_name}"
    run_dir = cfg.training.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_epoch_metrics(fold_dir: Path, metrics: list[dict[str, Any]]) -> None:
    save_json(fold_dir / "metrics.json", metrics)
    csv_lines = ["fold,epoch,train_mse,val_mse\n"]
    for item in metrics:
        csv_lines.append(
            f"{item['fold']},{item['epoch']},{item['train_mse']},{item['val_mse']}\n"
        )
    (fold_dir / "metrics.csv").write_text("".join(csv_lines), encoding="utf-8")


def train_one_epoch(
    loader: DataLoader,
    modules: dict[str, nn.Module],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg,
    device: torch.device,
    fold_index: int,
    epoch: int,
) -> float:
    for module in modules.values():
        module.train()

    use_amp = cfg.training.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total_loss = 0.0
    total_count = 0
    for batch_index, batch in enumerate(loader):
        if cfg.training.max_train_batches is not None and batch_index >= cfg.training.max_train_batches:
            break
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            prediction = forward_batch(batch, modules, cfg, device)
            loss = criterion(prediction, batch["target_delta"])
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * prediction.shape[0]
        total_count += prediction.shape[0]
        if cfg.training.log_every_n_steps and (
            (batch_index + 1) % cfg.training.log_every_n_steps == 0
            or batch_index == 0
        ):
            running_loss = total_loss / max(total_count, 1)
            print(
                f"fold={fold_index} epoch={epoch} step={batch_index + 1}/{len(loader)} "
                f"batch_mse={loss.item():.6f} running_mse={running_loss:.6f}"
            )

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(
    loader: DataLoader,
    modules: dict[str, nn.Module],
    criterion: nn.Module,
    cfg,
    device: torch.device,
) -> float:
    for module in modules.values():
        module.eval()

    total_loss = 0.0
    total_count = 0
    for batch_index, batch in enumerate(loader):
        if cfg.training.max_val_batches is not None and batch_index >= cfg.training.max_val_batches:
            break
        batch = move_batch_to_device(batch, device)
        prediction = forward_batch(batch, modules, cfg, device)
        loss = criterion(prediction, batch["target_delta"])
        total_loss += loss.item() * prediction.shape[0]
        total_count += prediction.shape[0]

    return total_loss / max(total_count, 1)


def make_optimizer(modules: dict[str, nn.Module], cfg) -> torch.optim.Optimizer:
    parameters = []
    for module in modules.values():
        parameters.extend(module.parameters())
    return torch.optim.AdamW(
        parameters,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )


def make_dataloader(dataset, cfg, device: torch.device, shuffle: bool) -> DataLoader:
    kwargs = {
        "batch_size": cfg.training.batch_size,
        "shuffle": shuffle,
        "num_workers": cfg.training.num_workers,
        "pin_memory": device.type == "cuda" and cfg.training.pin_memory,
    }
    if cfg.training.num_workers > 0:
        kwargs["prefetch_factor"] = cfg.training.prefetch_factor
        kwargs["persistent_workers"] = cfg.training.persistent_workers
    return DataLoader(dataset, **kwargs)


def run_training(
    smoke_test: bool = False,
    epochs: int | None = None,
    output_root: Path | None = None,
    experiment_name: str | None = None,
    num_folds: int | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    holdout_val_ratio: float | None = None,
    device_name: str | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    prefetch_factor: int | None = None,
    use_amp: bool | None = None,
    log_every_n_steps: int | None = None,
    model_name: str | None = None,
    history_frames: int | None = None,
    target_branch: str | None = None,
    context_branch: str | None = None,
    no_checkpoints: bool = False,
    no_feature_cache: bool = False,
    rebuild_feature_cache: bool = False,
) -> Path:
    cfg = default_config()
    if epochs is not None:
        cfg.training.epochs = epochs
    if output_root is not None:
        cfg.training.output_root = output_root
    if experiment_name is not None:
        cfg.training.experiment_name = experiment_name
    if num_folds is not None:
        cfg.training.num_folds = num_folds
    if max_train_batches is not None:
        cfg.training.max_train_batches = max_train_batches
    if max_val_batches is not None:
        cfg.training.max_val_batches = max_val_batches
    if holdout_val_ratio is not None:
        cfg.training.holdout_val_ratio = holdout_val_ratio
    if device_name is not None:
        cfg.training.device = device_name
    if batch_size is not None:
        cfg.training.batch_size = batch_size
    if num_workers is not None:
        cfg.training.num_workers = num_workers
    if prefetch_factor is not None:
        cfg.training.prefetch_factor = prefetch_factor
    if use_amp is not None:
        cfg.training.use_amp = use_amp
    if log_every_n_steps is not None:
        cfg.training.log_every_n_steps = log_every_n_steps
    if model_name is not None:
        cfg.data.model_name = model_name
    if history_frames is not None:
        cfg.data.history_frames = history_frames
    if target_branch is not None:
        cfg.data.target_branch = target_branch
    if context_branch is not None:
        cfg.data.context_branch = context_branch
    if no_checkpoints:
        cfg.training.save_checkpoints = False
    if no_feature_cache:
        cfg.training.use_feature_cache = False

    if smoke_test:
        cfg.training.epochs = 1
        cfg.training.max_train_batches = 1
        cfg.training.max_val_batches = 1

    set_seed(cfg.training.seed)
    device = resolve_device(cfg.training.device)
    print_preflight(device)
    print(
        "Training load config: "
        f"batch_size={cfg.training.batch_size}, "
        f"num_workers={cfg.training.num_workers}, "
        f"pin_memory={cfg.training.pin_memory}, "
        f"prefetch_factor={cfg.training.prefetch_factor}, "
        f"persistent_workers={cfg.training.persistent_workers}, "
        f"use_amp={cfg.training.use_amp}, "
        f"use_feature_cache={cfg.training.use_feature_cache}"
    )

    run_dir = create_run_dir(cfg)

    # Save the exact training config for later rollout/validation scripts.
    # 保存本次训练配置，后续独立 rollout/validation 会读取这个结果目录。
    save_json(run_dir / "config.json", cfg)
    print(f"Saving run outputs to: {run_dir}")

    # Load raw CalMS21 records and build asymmetric training windows.
    # 读取 CalMS21 原始记录，并构建非对称训练窗口。
    records = load_calms21_sequences(cfg.data.data_path)
    windows = build_window_indices(records, cfg.data)
    print(f"Loaded {len(records)} sequences and {len(windows)} windows.")

    feature_cache = None
    if cfg.training.use_feature_cache:
        print("Feature cache: checking/building raw window feature cache...")
        feature_cache = build_or_load_window_feature_cache(
            records,
            windows,
            cfg.data,
            rebuild=rebuild_feature_cache,
        )
        print(f"Feature cache: {feature_cache.status} at {feature_cache.path}")
    else:
        print("Feature cache: disabled")

    # Build train/validation folds by sequence_id to avoid leakage.
    # 按 sequence_id 构建训练/验证折，避免重叠窗口泄露。
    folds = build_kfold_splits(
        records,
        windows,
        cfg.training.split_mode,
        cfg.training.num_folds,
        cfg.training.seed,
        cfg.training.holdout_val_ratio,
    )

    all_fold_summaries: list[dict[str, Any]] = []
    for fold in folds:
        overlap = fold.train_sequence_ids & fold.val_sequence_ids
        if overlap and cfg.training.split_mode == "sequence_level":
            raise RuntimeError(f"Fold {fold.fold_index} has sequence leakage: {overlap}")

        fold_dir = run_dir / f"fold_{fold.fold_index + 1:02d}"
        checkpoint_dir = fold_dir / "checkpoints"
        fold_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"Fold {fold.fold_index + 1}/{len(folds)}: "
            f"train_windows={len(fold.train_windows)}, val_windows={len(fold.val_windows)}"
        )
        train_windows = fold.train_windows[:512] if smoke_test else fold.train_windows
        val_windows = fold.val_windows[:256] if smoke_test else fold.val_windows

        # Fit normalization statistics on the train fold only.
        # 只在当前训练折上拟合归一化统计量，避免验证集信息泄露。
        if feature_cache is not None:
            train_rows = feature_cache.row_indices(train_windows)
            val_rows = feature_cache.row_indices(val_windows)
            normalizers = fit_normalizers_from_cache(
                feature_cache,
                train_rows,
                cfg.data,
                cfg.normalization,
            )
            train_dataset = CachedCalMS21AsymmetricPoseDataset(
                feature_cache, train_rows, normalizers
            )
            val_dataset = CachedCalMS21AsymmetricPoseDataset(
                feature_cache, val_rows, normalizers
            )
        else:
            normalizers = fit_normalizers(
                records, train_windows, cfg.data, cfg.normalization
            )
            train_dataset = CalMS21AsymmetricPoseDataset(
                records, train_windows, cfg.data, normalizers
            )
            val_dataset = CalMS21AsymmetricPoseDataset(
                records, val_windows, cfg.data, normalizers
            )
        # DataLoader parallelism controls CPU-side loading pressure.
        # DataLoader 并行参数控制 CPU 侧数据加载压力。
        train_loader = make_dataloader(train_dataset, cfg, device, shuffle=True)
        val_loader = make_dataloader(val_dataset, cfg, device, shuffle=False)

        modules = build_modules(cfg, device)
        criterion = build_loss("mse").to(device)
        optimizer = make_optimizer(modules, cfg)

        fold_metrics: list[dict[str, Any]] = []
        best_val_loss = float("inf")
        for epoch in range(cfg.training.epochs):
            train_loss = train_one_epoch(
                train_loader,
                modules,
                criterion,
                optimizer,
                cfg,
                device,
                fold.fold_index + 1,
                epoch + 1,
            )
            val_loss = evaluate(val_loader, modules, criterion, cfg, device)
            epoch_metrics = {
                "fold": fold.fold_index + 1,
                "epoch": epoch + 1,
                "train_mse": train_loss,
                "val_mse": val_loss,
            }
            fold_metrics.append(epoch_metrics)
            save_epoch_metrics(fold_dir, fold_metrics)
            print(
                f"fold={fold.fold_index + 1} epoch={epoch + 1} "
                f"train_mse={train_loss:.6f} val_mse={val_loss:.6f}"
            )

            if cfg.training.save_checkpoints:
                if cfg.training.save_last_checkpoint:
                    save_checkpoint(
                        checkpoint_dir / "last.pt",
                        modules,
                        optimizer,
                        cfg,
                        normalizers,
                        fold.fold_index + 1,
                        epoch + 1,
                        train_loss,
                        val_loss,
                    )
                if cfg.training.save_best_checkpoint and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        checkpoint_dir / "best.pt",
                        modules,
                        optimizer,
                        cfg,
                        normalizers,
                        fold.fold_index + 1,
                        epoch + 1,
                        train_loss,
                        val_loss,
                    )

        best_epoch = min(fold_metrics, key=lambda item: item["val_mse"])
        all_fold_summaries.append(
            {
                "fold": fold.fold_index + 1,
                "train_windows": len(train_windows),
                "val_windows": len(val_windows),
                "best_epoch": best_epoch["epoch"],
                "best_val_mse": best_epoch["val_mse"],
                "last_train_mse": fold_metrics[-1]["train_mse"],
                "last_val_mse": fold_metrics[-1]["val_mse"],
            }
        )
        save_json(fold_dir / "summary.json", all_fold_summaries[-1])
        save_json(run_dir / "summary.json", all_fold_summaries)

        if smoke_test:
            break

    print(f"Run complete. Results saved to: {run_dir}")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one train batch and one validation batch for shape/loss checks.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count.")
    parser.add_argument("--num-folds", type=int, default=None, help="Override k-fold count.")
    parser.add_argument(
        "--holdout-val-ratio",
        type=float,
        default=None,
        help="Validation ratio used when --num-folds 1.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory where timestamped run folders are created.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name suffix for the timestamped run folder.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Optional training batch limit for quick trial runs.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Optional validation batch limit for quick trial runs.",
    )
    parser.add_argument("--device", type=str, default=None, help="Override device: auto/cpu/cuda.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader workers.")
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=None,
        help="Override DataLoader prefetch factor when num_workers > 0.",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Enable CUDA automatic mixed precision.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=None,
        help="Print batch-level loss every N training steps.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        choices=["single_predict", "multi_predict"],
        help="Select Single_predict or Multi_predict architecture.",
    )
    parser.add_argument(
        "--history-frames",
        type=int,
        default=None,
        help="Override the number of historical frames.",
    )
    parser.add_argument(
        "--target-branch",
        type=str,
        default=None,
        choices=["intruder", "resident"],
        help="Branch to predict.",
    )
    parser.add_argument(
        "--context-branch",
        type=str,
        default=None,
        choices=["intruder", "resident"],
        help="Context branch used by Multi_predict.",
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Save config and metrics only, without model checkpoint files.",
    )
    parser.add_argument(
        "--no-feature-cache",
        action="store_true",
        help="Disable raw window feature cache and use on-the-fly extraction.",
    )
    parser.add_argument(
        "--rebuild-feature-cache",
        action="store_true",
        help="Force rebuilding the raw window feature cache for this data/config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(
        smoke_test=args.smoke_test,
        epochs=args.epochs,
        output_root=args.output_root,
        experiment_name=args.experiment_name,
        num_folds=args.num_folds,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        holdout_val_ratio=args.holdout_val_ratio,
        device_name=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        use_amp=True if args.use_amp else None,
        log_every_n_steps=args.log_every_n_steps,
        model_name=args.model_name,
        history_frames=args.history_frames,
        target_branch=args.target_branch,
        context_branch=args.context_branch,
        no_checkpoints=args.no_checkpoints,
        no_feature_cache=args.no_feature_cache,
        rebuild_feature_cache=args.rebuild_feature_cache,
    )
