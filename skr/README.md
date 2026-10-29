# 自有数据集训练与视频物品检测

使用自己的图片和边界框标注训练 YOLO 检测模型，再对本地视频逐帧识别，输出带框视频与 CSV。当前提供命令行入口。

## 1. Python 环境

本机已有可用环境：Python 3.11、Ultralytics 8.4.184、PyTorch 2.5.1，CUDA 可用，显卡约 8GB 显存。

```bash
conda activate yolo
cd /home/bcw/Documents/Codex/2026-10-23/new-chat/skr
python train.py --help
```

如 conda 未初始化，可直接用 `/home/bcw/miniconda3/envs/yolo/bin/python` 替代下面命令中的 `python`。
本机默认 `python3` 指向另一个尚未安装训练依赖的 Python 3.14，请使用 yolo 环境。

新机器可创建 Python 3.11 环境并安装依赖：

```bash
conda create -n object-detection python=3.11 -y
conda activate object-detection
python -m pip install -r requirements.txt
```

需要 GPU 时按 PyTorch 官方安装页选择匹配的版本，再用 `python -c "import torch; print(torch.cuda.is_available())"` 确认。

## 2. 放入数据集

预留的目录结构如下：

```text
datasets/custom/
  images/train/     # 训练图片，例如 001.jpg
  images/val/       # 验证图片，例如 101.jpg
  labels/train/     # 同名标注，例如 001.txt
  labels/val/       # 同名标注，例如 101.txt
```

每张图片必须有同名 TXT，没有物品的背景图片使用空 TXT。每个物品一行，例如：

```text
0 0.5 0.5 0.4 0.3
1 0.2 0.3 0.1 0.2
```

依次为 `类别编号 中心x 中心y 宽 高`。类别从 0 开始；坐标与尺寸除以图片宽高，归一化到 0~1，框不得超出图片。
修改 `data.yaml` 中的 `names`，使类别顺序与编号一致。数据也可放在其他位置，只需修改 `path` 或指定 `--data`。相对路径以 YAML 所在位置为准。

建议按约 8:2 划分训练和验证集；同一段视频的相邻帧应放入同一集合，避免验证结果虚高。检查器拒绝训练和验证中内容完全相同的图片。两组都应覆盖所有类别。

只有图片、没有边界框标注时，需先使用标注工具画框并导出 YOLO 目标检测格式。分类文件夹、COCO JSON、VOC XML、分割多边形不能直接用于本项目，需要转换。

```bash
python dataset.py --data data.yaml
```

检查器校验图片、标注缺失、类别、坐标、数据重叠并输出图片与物品数量。示例目录为空时，检查失败是预期行为。

## 3. 训练模型

```bash
python train.py --data data.yaml --model yolo11n.pt --epochs 100 --imgsz 640 --batch 8 --device 0
```

第一次使用 `yolo11n.pt` 会下载预训练权重，需要联网。离线时指定已有的本地模型路径。
默认使用预训练模型微调；从随机权重开始训练可用 `--model yolo11n.yaml`，通常需要更多数据和训练时间。

`--device auto` 自动选择 CUDA 或 CPU，`--device cpu` 使用 CPU。显存不足先把 `--batch` 改为 4 或 2，必要时减小 `--imgsz`。多进程报错可加 `--workers 0`。

最优权重通常在 `runs/train/custom/weights/best.pt`，最后一轮权重在 `last.pt`。重复训练会生成 `custom2` 等新目录，以结束时打印的路径为准。验证指标和训练图位于同一目录。训练轮数不保证识别质量，需要结合验证集精度和漏检情况评估。

## 4. 导入视频并识别

把视频放入 `videos/`，或通过 `--source` 指定其他本地路径。路径含空格时加引号。

```bash
python predict_video.py --weights runs/train/custom/weights/best.pt --source videos/input.mp4 --output outputs/result.mp4 --device 0
```

输出 `outputs/result.mp4` 和 `outputs/result.csv`。CSV 包含从 0 开始的帧编号、按帧率计算的时间、类别、置信度和像素框坐标。一帧有多个物体会有多行，无目标帧没有记录。检测次数会重复计算不同帧的同一物体，未实现跟踪或去重计数。

`--conf 0.4` 提高显示阈值，`--show` 打开预览，按 Q 提前结束并保留已处理部分。默认不打开窗口。输出文件已存在时会拒绝覆盖，请更换 `--output`。

输出使用 OpenCV 的 mp4v 编码，**不保留音轨**。时间按固定帧率估算，可变帧率视频有同步需求时请先转换。输入须能被本机 OpenCV 解码，输出要求偶数宽高。模型只能识别训练过的类别。

## 5. 运行检查

```bash
python -m unittest discover -s tests -v
```

测试验证数据集检查逻辑，不下载模型。流程验证使用临时合成数据训练一轮并处理测试视频，只证明流程能运行，不代表真实物品识别精度。

完整流程验证可运行 `python tests/smoke_pipeline.py`，结束后自动清理临时合成数据与权重。

分发或商用前请查看 Ultralytics 官方许可证要求。
