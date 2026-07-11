# CalMS21 非对称姿态预测训练流程设计说明

本文档记录本轮讨论形成的训练方案和关键约束，供后续任务读取和继续开发。

## 1. 实验目标

当前任务不是行为分类，而是 **预测 B 鼠当前姿态**。

对每个目标时刻 `t`：

```text
A 输入: A[t-9 : t]       # A 鼠历史 9 帧 + 当前 1 帧，共 10 帧
B 输入: B[t-4 : t-1]     # B 鼠只提供历史 4 帧，不提供当前帧
目标:   B[t] - B[t-1]     # 预测 B 鼠当前姿态相对上一帧的位移
```

目标输出维度为 `14`，对应 B 鼠 7 个 joint 的 `(x, y)` 位移。

这个任务是一个条件预测任务：允许使用 `A[t]`，但不允许使用 `B[t]` 的姿态作为输入。

## 2. 数据来源

当前训练数据来自：

```text
Caltech/calms21_task1_train_distance_lt_330.npy
```

该文件由：

```text
Caltech/caltech_preprocess.py
```

从原始 `calms21_task1_train.npy` 生成。预处理逻辑是保留两只鼠中心点距离 `< 330` 像素的帧。

数据结构大致为：

```python
{
  "annotator-id_0": {
    "task1/train/mouse001_task1_annotator1": {
      "keypoints": ...,
      "scores": ...,
      "annotations": ...,
      "metadata": ...
    },
    ...
  }
}
```

关键字段：

```text
keypoints:   shape = (frames, 2, 2, 7)
scores:      shape = (frames, 2, 7)
annotations: shape = (frames,)
```

`keypoints` 索引含义：

```text
keypoints[frame, mouse_id, coord, joint_id]
```

其中：

```text
mouse_id = 0: A / resident
mouse_id = 1: B / intruder
coord = 0: x
coord = 1: y
joint_id = 0..6
```

行为标签 vocab：

```text
attack:        0
investigation: 1
mount:         2
other:         3
```

## 3. 模块化目录结构

训练流程拆分为多个文件，而不是全部写在一个大文件中：

```text
Model/asymmetric_pose_case/
  config.py
  data.py
  embeddings.py
  sequence_models.py
  pooling.py
  heads.py
  losses.py
  kfold.py
  train.py
  DESIGN_NOTES.md
```

各文件职责：

```text
config.py          集中 dataclass 配置
data.py            数据读取、窗口构建、特征计算、归一化
embeddings.py      embedding 子模块
sequence_models.py LSTM/GRU 等时序模型接口
pooling.py         last / mean / attention pooling
heads.py           prediction head
losses.py          loss 构建
kfold.py           k-fold 划分
train.py           串联完整训练流程
```

`train.py` 只负责显式串联流程：

```text
config
  -> data
  -> k-fold split
  -> embedding
  -> sequence model
  -> pooling
  -> prediction head
  -> loss
  -> optimizer
```

## 4. 配置化要求

实现必须是 configuration-driven。

所有关键参数都应集中在 `config.py` 的 dataclass 中，而不是散落硬编码在模块内部。

包括但不限于：

```text
A history length
B history length
joint embedding dim
pose embedding dim
velocity embedding dim
behavior embedding dim
mouse embedding dim
interaction embedding dim
sequence hidden dim
prediction head hidden dims
pooling type
behavior_label_mode
use_scores
num_folds
split_mode
batch_size
learning_rate
epochs
```

这样后续做 ablation study 或替换模块时，只需要改配置或替换对应模块。

## 5. Embedding 输入特征

当前确认的 embedding 输入特征包括：

```text
1. 每只鼠 7 个 joint 的 xy 坐标
2. 每只鼠 7 个 joint 的速度
3. 两只鼠中心点差值 B_center - A_center
4. 两只鼠中心点距离 ||B_center - A_center||
5. 两只鼠面朝方向夹角
6. annotation 信息
```

### 5.1 Joint xy

每只鼠每一帧有 7 个 joint，每个 joint 是 `(x, y)`。

在 embedding 中：

```text
JointEncoder: (x, y) -> joint latent
PoseEncoder:  7 个 joint latent -> pose embedding
```

`JointEncoder` 在所有 joint、A 鼠、B 鼠之间共享参数。

### 5.2 Joint velocity

每个 joint 的速度使用相邻帧差值：

```text
velocity[k] = xy[k] - xy[k-1]
```

对于 B 分支，最后一个可用时刻是 `t-1`，因此：

```text
B 在 t-1 的速度 = B[t-1] - B[t-2]
```

这样不会泄露目标帧 `B[t]`。

对于 A 分支，因为输入包含 `A[t]`，所以可以使用：

```text
A 在 t 的速度 = A[t] - A[t-1]
```

### 5.3 Center delta 和 center distance

每只鼠中心点定义为该鼠 7 个 joint 坐标的平均值：

```text
center = mean(joint xy over 7 joints)
```

交互特征包括：

```text
center_delta = B_center - A_center
center_distance = ||center_delta||
```

中心差值和 joint xy 存在信息冗余，但它能显式提供两只鼠相对位置，是有用的 inductive bias。

### 5.4 面朝方向夹角

朝向向量使用：

```text
heading = nose - neck
```

其中：

```text
nose joint index = 0
neck joint index = 3
```

两只鼠的朝向夹角不直接用角度值，而是编码为：

```text
sin(delta_angle)
cos(delta_angle)
```

这样可以避免角度在 `pi` 和 `-pi` 附近发生数值跳变。

### 5.5 Annotation

`annotations` 是全局交互/行为标签，不是 A 鼠或 B 鼠各自的私有标签。

annotation 使用 `nn.Embedding` 编码，而不是作为连续数值输入。

配置项：

```text
behavior_label_mode = "history_plus_current" | "history_only" | "none"
```

含义：

```text
history_plus_current: 历史 annotation 和当前 annotation 都可作为条件输入
history_only:         只使用历史 annotation
none:                 不使用 annotation
```

当前默认使用：

```text
behavior_label_mode = "history_plus_current"
```

此时当前 annotation 是一个 global condition，会在最终 head 前拼接，而不是作为 A 或 B 的私有特征。

## 6. 暂不使用 scores

`scores` 是姿态检测器给每个 keypoint 的置信度，shape 为：

```text
(frames, 2, 7)
```

它表示每个关键点坐标的检测可信程度。

当前第一版不使用 `scores`。

后续可以考虑两种扩展：

```text
1. 将 joint 输入从 (x, y) 改为 (x, y, score)
2. 新增独立 ScoreEncoder，再与 pose / velocity / annotation 融合
```

## 7. Embedding 层级结构

当前 embedding 设计为：

```text
Joint -> Pose -> Mouse
Velocity Joint -> Velocity -> Mouse
Annotation -> Mouse / Global condition
Interaction -> Interaction Embedding
```

鼠标分支：

```text
joint xy
  -> JointEncoder
  -> PoseEncoder

joint velocity
  -> VelocityJointEncoder
  -> VelocityEncoder

annotation
  -> BehaviorEmbedding

pose + velocity + behavior
  -> MouseEncoder
  -> Mouse Embedding
```

交互分支：

```text
center delta
center distance
facing angle sin/cos
  -> InteractionEncoder
  -> Interaction Embedding
```

A/B 共用同一套 mouse embedding 参数，确保两只鼠处于同一个 latent feature space。

Interaction 作为独立模态，不提前合并到任意一只鼠中。

## 8. 时序模型、Pooling 和 Head

当前默认时序模型：

```text
LSTM
```

A 分支和 B 分支共享 sequence model。

Interaction 分支使用独立 sequence branch。

流程：

```text
A mouse embeddings
  -> shared sequence model
  -> pooling

B mouse embeddings
  -> shared sequence model
  -> pooling

interaction embeddings
  -> interaction sequence model
  -> pooling

A pooled + B pooled + interaction pooled + optional current annotation embedding
  -> prediction head
  -> predicted B pose delta
```

Pooling 默认策略：

```text
LSTM: last valid hidden step
其他模型: mask mean pooling
```

后续可替换为 attention pooling。

Prediction Head 输出：

```text
14-dimensional normalized delta
```

Loss：

```text
MSELoss
```

Loss 模块保持可替换，后续可以换成 SmoothL1 / Huber / weighted loss。

## 9. K-fold 策略

默认使用：

```text
5-fold sequence-level cross validation
```

当前数据大约有：

```text
70 条 sequence
322,522 个窗口样本
```

5-fold 时，每折大约：

```text
14 条 sequence 做 validation
56 条 sequence 做 training
```

采用 sequence-level split 的原因：

```text
同一条 sequence 内的相邻窗口高度重叠。
如果按 window 随机划分，相邻窗口可能同时出现在 train 和 validation，
导致验证集结果偏乐观。
```

因此默认按 `sequence_id` 划分。

同时代码中保留 window-level split 选项，方便后续对比：

```text
split_mode = "sequence_level" | "window_level"
```

## 10. 归一化

连续变量使用 train fold 的 mean/std 做归一化。

需要归一化的内容：

```text
joint xy
joint velocity
interaction features
target delta
```

归一化统计量只在当前 fold 的训练集上拟合，避免验证集信息泄露。

## 11. 关键防泄露约束

必须保证：

```text
1. B[t] pose 只能作为 target，不能进入输入。
2. B[t-1] velocity = B[t-1] - B[t-2]，不能用 B[t] - B[t-1]。
3. 如果 behavior_label_mode = history_only，则不能输入当前 annotation。
4. 如果 behavior_label_mode = history_plus_current，则当前 annotation 是显式 global condition。
5. validation fold 的 normalization 不能参与统计量拟合。
6. sequence-level k-fold 下，同一个 sequence_id 不能同时出现在 train 和 validation。
```

## 12. 中英文注释要求

关键调用步骤需要添加简洁的中英文注释，说明该调用在做什么。

例如：

```python
# Build train/validation folds by sequence_id to avoid leakage.
# 按 sequence_id 构建训练/验证折，避免重叠窗口泄露。
folds = build_kfold_splits(...)

# Encode raw coordinates, velocities, and labels into latent tokens.
# 将原始坐标、速度和标签编码为 latent token。
embedded = embedder(batch)

# Predict B mouse current pose displacement B[t] - B[t-1].
# 预测 B 鼠当前姿态相对上一帧的位移 B[t] - B[t-1]。
prediction = prediction_head(...)
```

注释应解释关键数据流和设计决策，不需要给每一行代码都写注释。

## 13. 当前实现验证方式

快速 smoke test：

```powershell
python -m Model.asymmetric_pose_case.train --smoke-test
```

语法检查：

```powershell
python -m compileall Model\asymmetric_pose_case
```

完整训练：

```powershell
python -m Model.asymmetric_pose_case.train
```

当前 Windows Python 环境是 CPU-only PyTorch；如果在 Docker 中训练，需要确保 Docker 内部安装了 Python 和 CUDA 版 PyTorch，并在训练开始前检查：

```python
torch.cuda.is_available()
```
