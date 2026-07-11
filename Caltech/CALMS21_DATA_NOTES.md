# CalMS21 数据与预处理说明 / Data Notes

## 数据结构 / Data Format

`.npy` 文件加载方式：

```python
data = np.load("Caltech/calms21_task1_train.npy", allow_pickle=True).item()
```

顶层结构大致是：

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

每个实验序列里常见字段：

- `keypoints`: 小鼠姿态关键点，shape 为 `(frames, 2, 2, 7)`。
- `scores`: 每个关键点的检测置信度，shape 为 `(frames, 2, 7)`。
- `annotations`: 每帧行为标签 id，shape 为 `(frames,)`。
- `metadata`: 元信息，例如 `annotator-id` 和行为标签 vocab。

## Keypoints 含义 / Keypoint Meaning

`keypoints` 的索引方式：

```python
keypoints[frame, mouse_id, coord, keypoint_id]
```

各维度含义：

- `frame`: 帧编号。
- `mouse_id`: 小鼠编号，`0` 通常是 resident，`1` 通常是 intruder。
- `coord`: 坐标维度，`0` 是 x，`1` 是 y，单位是图像像素。
- `keypoint_id`: 每只小鼠身上的 7 个关键点。

7 个关键点大致为：

```text
0: nose / 鼻尖
1: ear / 一侧耳朵
2: ear / 另一侧耳朵
3: neck / 颈部或肩颈中心
4: hip / 一侧髋部
5: hip / 另一侧髋部
6: tail base / 尾根
```

中心点计算方式：

```python
center = np.nanmean(keypoints[frame, mouse_id], axis=1)
```

在 `Tools/Utils.py` 中已有封装：

```python
calculate_mouse_centers(keypoints_sequence)
calculate_mouse_distance(keypoints_sequence)
```

其中两只小鼠距离是每帧两个中心点之间的欧氏距离，单位是像素。

## 预处理脚本 / Preprocess Script

脚本位置：

```text
Caltech/caltech_preprocess.py
```

当前用途：读取一个 CalMS21 `.npy` 文件，计算每一帧两只小鼠中心点之间的距离，只保留距离小于阈值的帧。

当前默认配置在脚本顶部：

```python
INPUT_NPY_PATH = Path(__file__).resolve().parent / "calms21_task1_train.npy"
DISTANCE_LIMIT = 330
```

如果要处理另一个 `.npy` 文件，或修改距离阈值，直接改这两个常量。

筛选逻辑：

```python
distances = calculate_mouse_distance(sequence_data["keypoints"])
keep_mask = distances < DISTANCE_LIMIT
```

对每个实验序列：

- `keypoints` 按 `keep_mask` 筛选第一维。
- `scores` 按 `keep_mask` 筛选第一维。
- `annotations` 按 `keep_mask` 筛选第一维。
- `metadata` 原样保留。

如果某个实验序列没有任何帧满足条件，脚本仍然保留该序列，但对应数组长度会变成 `0`，并打印 warning 提醒。

## 输出文件名 / Output Filename

输出文件名会自动包含原始文件名和距离限制，例如：

```text
calms21_task1_train_distance_lt_330.npy
```

含义：

- `calms21_task1_train`: 来源文件是 `calms21_task1_train.npy`。
- `distance_lt_330`: 只保留两只小鼠中心距离 `< 330` 像素的帧。

处理后的 `.npy` 仍保持原始嵌套字典格式，方便下游代码继续按原格式读取。

