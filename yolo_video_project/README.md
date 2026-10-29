# 视频实时分割检测

## 直接运行

双击项目目录中的 `视频检测.py`，或者双击 `启动项目.sh`。

程序会自动：

1. 弹出视频选择窗口；
2. 加载训练好的分割权重；
3. 实时播放视频并在物体边缘绘制多边形轮廓；
4. 将结果保存到 `output/视频名_segment_时间/`。

运行窗口中按 `Q` 或 `Esc` 停止检测。

如果系统当前使用的是 `base` Python，脚本会自动切换到项目的 `yolo` 环境。也可以直接在终端运行：

```bash
python 视频检测.py
```

不需要输入视频路径，运行后在弹出的窗口中选择视频即可。

## 输出文件

每次检测会生成：

- `result.mp4`：绘制多边形边缘的视频；
- `detections.csv`：每个实例的类别、置信度和多边形点；
- `summary.json`：处理帧数、实例数和运行状态。

## 数据集分割

运行 `数据集分割.py`，按提示输入包含“图片 + 同名 LabelMe JSON”的目录、类别名称和输出目录。程序会按 80%/20% 划分，并生成 YOLO 分割格式：

```bash
python 数据集分割.py
```

也可以直接传入参数：

```bash
python 数据集分割.py --source "/你的原始数据集" --names "red metal bracket,blue bracket"
```

## 轮廓参数

默认轮廓简化值为 3 像素，可以直接修改 `video_segment.py` 中的默认值：

```python
default=3.0
```

数值越大，轮廓越平直，但细节越少。当前默认模型为：

```text
runs/segment/segment_20261027_205536_165577/weights/best.pt
```
