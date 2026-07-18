# CalMS21 Asymmetric Pose Prediction Design Notes

This document records the current design of `Model/asymmetric_pose_case` so future
changes can stay aligned with the implemented training path.

## 1. Task Goal

The current task is pose delta prediction, not behavior classification.

For each target time `t`, the model predicts the current pose displacement of the
configured target branch:

```text
target_delta = target_pose[t] - target_pose[t - 1]
```

Default branch configuration:

```text
target_branch  = "intruder"
context_branch = "resident"
```

The output dimension is `14`, corresponding to 7 joints times `(x, y)`.

The model supports two architecture modes:

```text
single_predict: use target branch history plus optional current behavior condition
multi_predict:  additionally use context branch and interaction sequence
```

The default is:

```text
model_name = "multi_predict"
```

## 2. Data Source

Current training data comes from the windowed preprocessed file:

```text
Caltech/calms21_task1_train_windowed_distance_lt_330.npy
```

This file is already transformed into trial windows and filtered by mouse distance.
The loaded Python object is keyed by `sequence_id`:

```python
{
  "task1/train/mouse001_task1_annotator1": [
    [
      [intruder_frame_0, intruder_frame_1, ...],
      [resident_frame_0, resident_frame_1, ...],
    ],
    ...
  ],
  ...
}
```

Branch index convention:

```text
0: intruder
1: resident
```

Each frame contains pose, velocity, self-distance, and interaction metadata:

```text
body.node.<joint>.position
body.node.<joint>.node_velocity
body.internode_distance
interaction.annotation_tag
interaction.intruder_or_resident_tag
interaction.mouse_midpoint_distance
interaction.head_angle
```

The directory `Caltech/task1_classic_classification/` comes from the official
CalMS21 task1 behavior-classification starter kit. It is a source artifact name,
not the current training objective.

## 3. Module Layout

The package is intentionally split into small modules:

```text
config.py          dataclass configuration
data.py            data loading, window indices, raw feature cache, datasets, normalization
embeddings.py      target/context/interaction embedding modules
sequence_models.py LSTM/GRU sequence model factory
pooling.py         last/mean/attention-style pooling
heads.py           pose delta prediction head
losses.py          loss factory
kfold.py           train/validation split construction
runtime.py         runtime helpers, checkpoint IO, device movement
train.py           end-to-end training orchestration
```

High-level training flow:

```text
config
  -> load records
  -> build full window index list
  -> build/load raw feature cache
  -> build train/validation split
  -> fit fold normalizers from train rows only
  -> cached train/validation datasets
  -> model / loss / optimizer
  -> train, evaluate, checkpoint
```

## 4. Configuration

Important `DataConfig` fields:

```text
data_path
feature_cache_dir = "Caltech/cache"
model_name
target_branch
context_branch
history_frames
include_context_current
include_interaction_current
behavior_label_mode
use_scores
```

Important `TrainingConfig` fields:

```text
num_folds = 1
split_mode = "sequence_level"
holdout_val_ratio = 0.2
batch_size
epochs
learning_rate
num_workers
pin_memory
prefetch_factor
persistent_workers
use_amp
use_feature_cache = True
save_checkpoints
```

CLI overrides are exposed in `train.py`, including:

```text
--device
--batch-size
--num-workers
--prefetch-factor
--use-amp
--num-folds
--holdout-val-ratio
--no-feature-cache
--rebuild-feature-cache
--no-checkpoints
```

## 5. Window Construction

`build_window_indices(records, data_config)` creates a flat list of `WindowIndex`:

```text
sequence_index
window_index
target_t
```

The minimum valid `target_t` is:

```text
history_frames
```

Default temporal lengths:

```text
target branch history:      history_frames
context branch sequence:    history_frames + include_context_current
interaction sequence:       history_frames + include_interaction_current
```

With the current defaults:

```text
history_frames = 9
include_context_current = True
include_interaction_current = True
```

The logs from the current dataset show roughly:

```text
70 sequences
307811 windows
```

## 6. Raw Feature Cache

The current training path uses an automatic raw feature cache to avoid repeatedly
extracting nested Python `dict/list` window features inside every DataLoader worker.

Default cache root:

```text
Caltech/cache
```

Cache directory format:

```text
Caltech/cache/asymmetric_pose_<hash>/
  manifest.json
  a_xy.npy
  a_velocity.npy
  a_self_distance.npy
  a_behavior.npy
  a_role.npy
  b_xy.npy
  b_velocity.npy
  b_self_distance.npy
  b_behavior.npy
  b_role.npy
  interaction.npy
  current_behavior.npy
  target_delta.npy
  previous_pose.npy
  target_pose.npy
  target_t.npy
  sequence_index.npy
  window_index.npy
```

The cache stores raw, unnormalized features. Fold-specific normalization is still
fit later using only the current fold's train rows.

The manifest/key includes:

```text
preprocess_version
source data path
source file size
source mtime
target_branch
context_branch
history_frames
include_context_current
include_interaction_current
num_joints
coord_dim
num_self_distances
sequence_ids
```

Cache behavior:

```text
use_feature_cache=True: build/load cache automatically
--no-feature-cache:     use old on-the-fly extraction dataset
--rebuild-feature-cache: force rebuild for the current data/config key
```

The cache is built once before k-fold training. Each fold maps its
`train_windows` and `val_windows` to cache row indices, so k-fold does not repeat
raw feature extraction for the same window.

## 7. Feature Semantics

Target branch features:

```text
a_xy
a_velocity
a_self_distance
a_behavior
a_role
```

Context branch features:

```text
b_xy
b_velocity
b_self_distance
b_behavior
b_role
```

Interaction features are 7-dimensional per frame:

```text
context_center - target_center
mouse_midpoint_distance
sin(target_head_angle)
cos(target_head_angle)
sin(context_head_angle)
cos(context_head_angle)
```

Behavior labels come from `interaction.annotation_tag` using `argmax`.
Role labels come from `interaction.intruder_or_resident_tag`.

The current implementation allows `include_interaction_current=True`, which means
the interaction sequence includes the target time `t`. This is a conditional
prediction design choice in the current code and should be reviewed carefully
before treating validation metrics as pure future-pose forecasting metrics.

## 8. Embedding, Sequence Model, Pooling, Head

Embedding hierarchy:

```text
joint xy          -> JointEncoder -> PoseEncoder
joint velocity    -> VelocityJointEncoder -> VelocityEncoder
self distance     -> SelfDistanceEncoder
behavior label    -> BehaviorEmbedding
role label        -> RoleEmbedding
mouse features    -> MouseEncoder
interaction       -> InteractionEncoder
```

Target and context branches share the same mouse embedding and temporal model
weights so both branches live in the same latent space.

Default temporal model:

```text
LSTM
```

Default multi-predict flow:

```text
target mouse embeddings
  -> shared mouse sequence model
  -> mouse pooler

context mouse embeddings
  -> shared mouse sequence model
  -> mouse pooler

interaction embeddings
  -> interaction sequence model
  -> interaction pooler

target pooled + context pooled + interaction pooled + optional current behavior
  -> prediction head
  -> normalized target_delta prediction
```

Loss:

```text
MSELoss
```

## 9. Train/Validation Splits

Supported split modes:

```text
sequence_level
window_level
```

Default:

```text
split_mode = "sequence_level"
num_folds = 1
holdout_val_ratio = 0.2
```

When `num_folds == 1`, the code creates a single holdout validation split using
`holdout_val_ratio`; it does not put all windows into validation.

When `num_folds > 1`, sequence-level k-fold shuffles sequence indices using the
configured seed and assigns each fold's validation sequences by round-robin.

For sequence-level split, a `sequence_id` must not appear in both train and
validation for the same fold.

Window-level split is retained for experiments, but it can leak highly overlapping
neighbor windows between train and validation and should not be used for the main
reported metrics.

## 10. Normalization

Continuous features are normalized using mean/std fit on the current fold's
training rows only.

Normalized groups:

```text
coord:          a_xy, b_xy
velocity:       a_velocity, b_velocity
self_distance:  a_self_distance, b_self_distance
interaction:    interaction
target_delta:   target_delta
```

Validation rows never contribute to normalizer statistics.

With feature cache enabled, normalizers are fit by `fit_normalizers_from_cache()`
using chunked row reads to avoid materializing the whole fold at once.

## 11. Performance Notes

The raw feature cache removes the original bottleneck of repeatedly parsing nested
window dictionaries inside DataLoader workers.

Observed current bottlenecks after cache:

```text
large batch + many workers can exhaust Docker /dev/shm
default_collate still stacks many per-sample tensors
worker shared-memory transfer can bottleneck before GPU compute
```

Stable CUDA starting point:

```bash
python -m Model.asymmetric_pose_case.train \
  --device cuda \
  --batch-size 2048 \
  --num-workers 4 \
  --prefetch-factor 1 \
  --use-amp
```

Larger batches reduce `steps_per_epoch`:

```text
steps_per_epoch ~= train_windows / batch_size
```

For fair comparisons, scale epochs with batch size if the goal is to keep optimizer
update count similar:

```text
1024 batch, 5 epochs   ~= baseline update count
2048 batch, 10 epochs  ~= similar updates
4096 batch, 20 epochs  ~= similar updates
8192 batch, 40 epochs  ~= similar updates
```

If using Docker, increase shared memory for aggressive DataLoader settings:

```bash
docker run --shm-size=8g ...
# or
docker run --ipc=host ...
```

Future performance improvement: implement a batch-level cached loader or custom
collate path that slices cache arrays by a whole batch of row indices at once,
rather than building and collating many per-sample tensors.

## 12. Checkpoints

`save_checkpoint()` currently writes:

```text
fold
epoch
train_mse
val_mse
config
normalizers
module state_dicts
optimizer state_dict
```

The checkpoint files are saved under:

```text
<run_dir>/fold_XX/checkpoints/last.pt
<run_dir>/fold_XX/checkpoints/best.pt
```

On Docker bind mounts backed by Windows, `torch.save()` can occasionally fail when
opening or replacing `.pt` files. If checkpoint IO fails but training itself is
stable, use `--no-checkpoints` to confirm the issue is IO-only, or write
`--output-root` to a native Linux path such as `/tmp/asymmetric_pose_runs`.

Recommended future hardening: save to a temporary checkpoint file first, then
atomically replace `last.pt` or `best.pt`; optionally fall back to legacy PyTorch
serialization if the default zip writer fails on a mounted filesystem.

## 13. Safety Constraints

Important invariants:

```text
normalization statistics must use train rows only
sequence-level split must not leak the same sequence_id into train and validation
raw feature cache must not store fold-normalized features
cache key must change when feature extraction semantics change
window-level split should not be used for final leakage-sensitive metrics
```

The current interaction-current design can include target-time interaction
features. Treat that as an explicit modeling choice; changing it requires updating
the cache preprocess version and rebuilding the cache.

## 14. Verification

Fast syntax check:

```bash
python -m py_compile \
  Model/asymmetric_pose_case/config.py \
  Model/asymmetric_pose_case/data.py \
  Model/asymmetric_pose_case/train.py
```

Fallback smoke test without cache:

```bash
python -m Model.asymmetric_pose_case.train \
  --smoke-test \
  --device cpu \
  --no-feature-cache \
  --no-checkpoints
```

Cached smoke test:

```bash
python -m Model.asymmetric_pose_case.train \
  --smoke-test \
  --device cpu \
  --no-checkpoints
```

CUDA training example:

```bash
python -m Model.asymmetric_pose_case.train \
  --device cuda \
  --batch-size 2048 \
  --num-workers 4 \
  --prefetch-factor 1 \
  --use-amp
```
