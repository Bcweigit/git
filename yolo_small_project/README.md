# 50 张图片起步的 YOLO 小项目

这个项目按以下流程组织：在 X-AnyLabeling 中人工标注约 50 张图片 → 训练 YOLO 小模型 → 用训练好的模型为其余图片生成预标注 → 在 X-AnyLabeling 中逐张校正。

## 1. 准备图片和类别

把所有待处理的原始图片放入 `images/all/`。先在 `classes.txt` 中填写类别名称，每行一个类别，名称必须和 X-AnyLabeling 中使用的标签完全一致。项目模板默认为 `object`，开始标注前请改成你的实际类别。

建议 50 张图片尽量覆盖不同场景、角度、大小、光照和遮挡情况。不要只选相似图片。标注时框住完整目标，避免漏标；同一类别要使用统一名称。

## 2. 在 X-AnyLabeling 手工标注

1. 打开 `images/manual50/` 中的图片目录进行标注（可以先从 `images/all/` 复制你挑选的 50 张图片进去）。
2. 把已标注图片和同名 `.json` 放到 `images/manual50/`。编辑 `classes.txt`，每行写一个真实类别，顺序必须与 X-AnyLabeling 导出 JSON 中的类别 ID 顺序一致。
3. 运行转换脚本，它会把框转换为 YOLO 检测格式（`class_id x_center y_center width height`，坐标归一化到 0–1），并自动按 80/20 划分训练集和验证集：

   ```bash
   python convert_xanylabeling.py
   ```

   如果原图和 JSON 在别的目录，可以传入目录路径：`python convert_xanylabeling.py /图片和JSON所在目录`。转换结果会放入 `dataset/images/{train,val}` 和 `dataset/labels/{train,val}`。

目录示例：

```text
dataset/
  images/train/a.jpg
  images/val/b.jpg        # 可选
  labels/train/a.txt
  labels/val/b.txt        # 可选
```

图片与标签文件必须同名（扩展名不同）。JSON 不能只改扩展名为 `.txt`，请使用上面的转换脚本。

## 3. 配置类别并训练

确认 `classes.txt` 与 `data.yaml` 中的类别名称和顺序一致，然后运行：

```bash
conda activate yolo
cd ~/Documents/Codex/2026-10-23/new-chat/yolo_small_project
bash train.sh
```

默认模型是 `yolo26n.pt`（轻量版），默认 100 个 epoch、图像边长 640。若希望先快速试跑，可执行：

```bash
EPOCHS=5 bash train.sh
```

如果 `yolo26n.pt` 首次下载不方便，可以把 `MODEL=yolo11n.pt` 传给脚本。第一次训练需要能访问权重下载源。结果保存在 `runs/detect/train/`；最佳权重通常为 `runs/detect/train/weights/best.pt`。

## 4. 给剩余图片生成预标注

把尚未标注的图片放进 `images/unlabeled/`，然后运行：

```bash
bash predict.sh runs/detect/train/weights/best.pt
```

脚本会用模型推理并将预测框保存为 YOLO 文本到 `prelabels/labels/`，同时复制图片到 `prelabels/images/`，便于核对同名文件。置信度阈值默认为 0.25，可这样调整：

```bash
CONF=0.35 bash predict.sh runs/detect/train/weights/best.pt
```

## 5. 在 X-AnyLabeling 中复核

将 `prelabels/images/` 作为待标注图片目录打开，并使用 `prelabels/labels/` 中对应的 YOLO 标签作为预标注导入/自动标注结果（不同版本入口名称略有差异）。重点检查漏检、错框和类别错误。模型预标注只是草稿，复核完成后再导出最终数据集。

## 将自训权重加载到 X-AnyLabeling

X-AnyLabeling 4.2.0 的 YOLO26 自动标注后端通过 ONNX Runtime 加载 ONNX 模型，不直接加载 Ultralytics `.pt`。在 `yolo` 环境中安装导出依赖并导出：

```bash
conda activate yolo
cd ~/Documents/Codex/2026-10-23/new-chat/yolo_small_project
python -m pip install onnx onnxslim
bash export_for_xanylabeling.sh
```

然后在 X-AnyLabeling 的自动标注模型菜单选择 **Custom → Load custom model**，打开本目录的 `xanylabeling_model.yaml`。配置里的类别由 `classes.txt` 生成；更新类别后请同步更新 YAML。确认模型加载成功后打开待标注图片，点自动标注运行按钮，并人工检查、保存预测框。

## 常见检查

- `classes.txt` 的顺序必须与 YOLO 标签中的 class_id 对应：第一行是 0，第二行是 1，以此类推。
- 训练前必须保证每个图片的 `.txt` 标签同名存在；负样本图片可使用空 `.txt` 文件。
- 50 张对训练小模型来说较少，结果只适用于相近场景。预标注后认真修正，再用修正后的更多数据继续训练。
- 若 CUDA 检查显示不可用，先确认在 `yolo` 环境中运行，并用 `python -c "import torch; print(torch.cuda.is_available())"` 检查。
