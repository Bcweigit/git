import os
import shutil
import random

# 图片目录
RAW_IMG = "dataset/parts_raw"
# 全部txt标签目录
RAW_LABEL = "dataset/labels_all"

# 输出目录
IMG_TRAIN = "dataset/images/train"
IMG_VAL = "dataset/images/val"
LABEL_TRAIN = "dataset/labels/train"
LABEL_VAL = "dataset/labels/val"

VAL_RATE = 0.2

# 创建文件夹
for d in [IMG_TRAIN, IMG_VAL, LABEL_TRAIN, LABEL_VAL]:
    os.makedirs(d, exist_ok=True)

# 读取图片
exts = (".jpg", ".jpeg", ".png", ".bmp")
img_list = [f for f in os.listdir(RAW_IMG) if f.lower().endswith(exts)]
random.shuffle(img_list)

split_idx = int(len(img_list) * (1 - VAL_RATE))
train_imgs = img_list[:split_idx]
val_imgs = img_list[split_idx:]

def copy_pair(img_filename, dst_img, dst_label):
    base = os.path.splitext(img_filename)[0]
    src_img = os.path.join(RAW_IMG, img_filename)
    src_txt = os.path.join(RAW_LABEL, base + ".txt")
    shutil.copy(src_img, dst_img)
    if os.path.exists(src_txt):
        shutil.copy(src_txt, dst_label)
    else:
        print(f"⚠️ 找不到标签文件: {base}.txt")

# 复制训练集
for name in train_imgs:
    copy_pair(name, IMG_TRAIN, LABEL_TRAIN)
# 复制验证集
for name in val_imgs:
    copy_pair(name, IMG_VAL, LABEL_VAL)

print(f"✅ 划分完成！\n训练集：{len(train_imgs)} 张\n验证集：{len(val_imgs)} 张")
