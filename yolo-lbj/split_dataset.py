import os
import shutil
import random

# ========= 路径（已经适配你的目录，不用改！） =========
RAW_FOLDER = "dataset/parts_raw"
IMG_TRAIN = "dataset/images/train"
IMG_VAL = "dataset/images/val"
LABEL_TRAIN = "dataset/labels/train"
LABEL_VAL = "dataset/labels/val"

VAL_RATE = 0.2  # 20%给验证集

# 创建文件夹（不存在自动建）
for d in [IMG_TRAIN, IMG_VAL, LABEL_TRAIN, LABEL_VAL]:
    os.makedirs(d, exist_ok=True)

# 读取图片
suffixs = (".jpg", ".jpeg", ".png", ".bmp")
img_list = [f for f in os.listdir(RAW_FOLDER) if f.lower().endswith(suffixs)]
random.shuffle(img_list)

split_point = int(len(img_list) * (1 - VAL_RATE))
train_imgs = img_list[:split_point]
val_imgs = img_list[split_point:]

def copy_one(img_filename, dst_img, dst_label):
    base_name = os.path.splitext(img_filename)[0]
    src_img = os.path.join(RAW_FOLDER, img_filename)
    src_txt = os.path.join(RAW_FOLDER, base_name + ".txt")
    shutil.copy(src_img, dst_img)
    if os.path.exists(src_txt):
        shutil.copy(src_txt, dst_label)

# 复制训练集
for name in train_imgs:
    copy_one(name, IMG_TRAIN, LABEL_TRAIN)
# 复制验证集
for name in val_imgs:
    copy_one(name, IMG_VAL, LABEL_VAL)

print(f"✅ 划分完成！\n训练集：{len(train_imgs)} 张\n验证集：{len(val_imgs)} 张")
