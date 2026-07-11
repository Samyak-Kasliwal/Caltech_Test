# Validation Rollout 与可视化说明

本文档记录 `Model/asymmetric_pose_case` 中 validation/rollout 的设计和使用方式，方便后续任务或其他 agent 直接读取。

## 1. 训练与 Validation 解耦

当前流程分成两步：

```text
train.py
  只负责训练模型、保存配置、保存指标、保存 checkpoint

validation/rollout.py
  单独加载训练结果，执行连续多帧预测，并输出可视化所需数据
```

这样做的原因是：

- 训练逻辑和验证/可视化逻辑互不耦合。
- 可以先训练一个模型，再用不同 `sequence_id`、`start_t`、`rollout_length` 多次做 rollout。
- rollout 长度不是固定的 20 帧，20f 只是默认示例。

## 2. 模型预测目标

模型内部预测的是 B 鼠当前姿态相对上一帧的位移：

```text
B_pred_delta[u] = B_pred[u] - B_history_last
```

但是 rollout 输出给可视化使用的是绝对坐标：

```text
B_pred_pose_xy[u] = B_history_last + B_pred_delta[u]
```

因此，可视化时应优先使用：

```text
b_pred_pose_xy
```

而不是直接使用 `b_pred_delta_xy`。

## 3. 连续 Rollout 逻辑

对第 `s` 步预测：

```text
u = start_t + s
```

模型输入窗口为：

```text
A input: A[u-9 : u]
B input: B[u-4 : u-1]
interaction: u-4 : u-1
annotation: annotation[u]
```

关键约定：

- A 分支使用真实 A 帧，包含未来 A，因为 A 被视为已知条件。
- annotation 使用真实全局 annotation，作为已知条件。
- B 分支在 `start_t` 前使用真实 B 历史。
- 从第二步开始，B 历史会混入前面已经预测出的 B 姿态。
- 真实 B 只用于输出对照和计算误差，不进入模型输入。

递归过程：

```text
1. 构造当前 target_t 的 A/B/interaction/annotation 输入
2. 模型输出 normalized delta
3. 使用 checkpoint 中保存的 target_delta normalizer 反归一化
4. reshape 为 [7, 2]
5. 用上一帧 B 的绝对坐标重建当前 B 预测姿态
6. 将 B 预测姿态写回 rollout buffer
7. 下一帧继续使用该 buffer 构造 B 历史窗口
```

## 4. 关键代码位置

```text
Model/asymmetric_pose_case/runtime.py
```

训练和验证共用逻辑：

- 构建模型模块
- 执行 forward
- 保存/加载 checkpoint
- 恢复 normalizer

```text
Model/asymmetric_pose_case/validation/config.py
```

`RolloutConfig` 配置项，包含：

- `run_dir`
- `fold`
- `checkpoint_name`
- `sequence_id`
- `start_t`
- `rollout_length`
- `output_name`
- `use_true_a_future`
- `use_true_annotation_future`

```text
Model/asymmetric_pose_case/validation/rollout.py
```

连续预测入口，负责：

- 加载 checkpoint
- 找到指定 sequence
- 维护 B rollout buffer
- 输出 `.npz` 和 `.json`

```text
Model/asymmetric_pose_case/validation/metrics.py
```

计算每帧误差和每个 joint 的 L2 error。

## 5. 训练命令

推荐第一次正式训练使用：

```powershell
python -m Model.asymmetric_pose_case.train ^
  --epochs 10 ^
  --experiment-name baseline_lstm_10epochs
```

如果只想检查流程：

```powershell
python -m Model.asymmetric_pose_case.train ^
  --smoke-test ^
  --experiment-name pipeline_check
```

训练结果会保存到：

```text
Model/asymmetric_pose_case/runs/<timestamp>_<experiment_name>/
```

其中每个 fold 的 checkpoint 位于：

```text
fold_01/checkpoints/best.pt
fold_01/checkpoints/last.pt
```

## 6. Rollout 命令

做 20 帧连续预测：

```powershell
python -m Model.asymmetric_pose_case.validation.rollout ^
  --run-dir Model/asymmetric_pose_case/runs/<your_run_dir> ^
  --fold 1 ^
  --checkpoint-name best.pt ^
  --sequence-id task1/train/mouse001_task1_annotator1 ^
  --start-t 100 ^
  --rollout-length 20
```

做 50 帧连续预测：

```powershell
python -m Model.asymmetric_pose_case.validation.rollout ^
  --run-dir Model/asymmetric_pose_case/runs/<your_run_dir> ^
  --fold 1 ^
  --checkpoint-name best.pt ^
  --sequence-id task1/train/mouse001_task1_annotator1 ^
  --start-t 100 ^
  --rollout-length 50
```

输出目录：

```text
Model/asymmetric_pose_case/runs/<your_run_dir>/fold_01/rollouts/
```

## 7. Rollout 输出文件

rollout 会生成：

```text
rollout_predictions.npz
rollout_summary.json
```

`rollout_predictions.npz` 中的主要数组：

```text
a_pose_xy
  shape: [rollout_length, 7, 2]
  含义: A 鼠真实绝对坐标

b_pred_pose_xy
  shape: [rollout_length, 7, 2]
  含义: B 鼠预测绝对坐标

b_true_pose_xy
  shape: [rollout_length, 7, 2]
  含义: B 鼠真实绝对坐标

b_pred_delta_xy
  shape: [rollout_length, 7, 2]
  含义: B 鼠预测位移

b_true_delta_xy
  shape: [rollout_length, 7, 2]
  含义: B 鼠真实位移

per_frame_mse
  shape: [rollout_length]
  含义: 每一帧 B 预测姿态的 MSE

per_frame_rmse
  shape: [rollout_length]
  含义: 每一帧 B 预测姿态的 RMSE

per_joint_l2
  shape: [rollout_length, 7]
  含义: 每一帧、每个 joint 的 L2 距离误差

target_t
  shape: [rollout_length]
  含义: 每个预测结果对应的原始帧号

keypoints_pred_pair
  shape: [rollout_length, 2, 2, 7]
  含义: 接近 CalMS21 原始 keypoints 的结构
```

`keypoints_pred_pair` 的结构为：

```text
keypoints_pred_pair[frame, mouse, coord, joint]

mouse = 0: A 真实姿态
mouse = 1: B 预测姿态
coord = 0: x
coord = 1: y
```

如果已有可视化函数吃 CalMS21 原始格式，优先使用：

```python
keypoints = data["keypoints_pred_pair"]
```

## 8. 读取 Rollout 结果示例

```python
import numpy as np

path = "Model/asymmetric_pose_case/runs/<your_run_dir>/fold_01/rollouts/rollout_predictions.npz"
data = np.load(path)

a_pose = data["a_pose_xy"]                # [T, 7, 2]
b_pred = data["b_pred_pose_xy"]           # [T, 7, 2]
b_true = data["b_true_pose_xy"]           # [T, 7, 2]
per_frame_rmse = data["per_frame_rmse"]   # [T]

# CalMS21-style keypoints, useful for existing visualization code.
# CalMS21 风格 keypoints，可直接给已有可视化函数使用。
keypoints_pred_pair = data["keypoints_pred_pair"]  # [T, 2, 2, 7]
```

## 9. 注意事项

- `start_t` 必须大于等于模型所需最小历史长度，目前至少为 `9`。
- `start_t + rollout_length` 不能超过该 sequence 的总帧数。
- rollout 当前假设真实未来 A 和真实未来 annotation 可用。
- 若未来要做完全闭环预测，需要额外预测 A 或设计没有未来 A/annotation 的配置。
- `b_pred_pose_xy` 是可视化主输出；`b_pred_delta_xy` 只是模型内部输出的分析形式。
