import os
import json
import numpy as np

# ========== 和标注里的类别名严格一致 ==========
json_folder = "dataset/parts_raw"
output_label_dir = "dataset/labels_all"
class_names = ["orange", "blue"]
# =============================================

class_map = {name: idx for idx, name in enumerate(class_names)}
os.makedirs(output_label_dir, exist_ok=True)

success_count = 0
empty_count = 0

for file in os.listdir(json_folder):
    if not file.endswith(".json"):
        continue
    
    json_path = os.path.join(json_folder, file)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    img_w = data["imageWidth"]
    img_h = data["imageHeight"]
    txt_name = file.replace(".json", ".txt")
    txt_path = os.path.join(output_label_dir, txt_name)

    lines = []
    for shape in data["shapes"]:
        label = shape["label"]
        if label not in class_map:
            continue
        
        cls_id = class_map[label]
        pts = np.array(shape["points"], dtype=np.float32)
        
        # 取所有点的极值，自动计算外接矩形
        x1 = np.min(pts[:, 0])
        y1 = np.min(pts[:, 1])
        x2 = np.max(pts[:, 0])
        y2 = np.max(pts[:, 1])
        
        # 归一化转YOLO格式
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        w = abs(x2 - x1) / img_w
        h = abs(y2 - y1) / img_h
        
        if w > 0.001 and h > 0.001:
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    
    if len(lines) > 0:
        success_count += 1
    else:
        empty_count += 1
        print(f"⚠️ 空标注文件：{file}")

print(f"\n转换完成：有效标注 {success_count} 个，空标注 {empty_count} 个")
print(f"输出路径：{output_label_dir}")
