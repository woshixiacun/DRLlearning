import cv2
import numpy as np
import os

# 1. 路径 ------------------------------------------------------------------
# img_path = r"/mnt/d/Study_File/codezcy/DRLlearning/ultralytics-main/datasets/coco8/images/train/000000000025.jpg"
# txt_path = r"/mnt/d/Study_File/codezcy/DRLlearning/ultralytics-main/datasets/coco8/labels/train/000000000025.txt"
# 1. 路径 ------------------------------------------------------------------
img_path = r"/mnt/d/Study_File/codezcy/DRLlearning/ultralytics-main/datasets/NEU_DET/mydata/images/val/scratches_37.jpg"
txt_path = r"/mnt/d/Study_File/codezcy/DRLlearning/ultralytics-main/datasets/NEU_DET/mydata/labels/val/scratches_37.txt"
save_path = os.path.join('/mnt/d/Study_File/codezcy', "000000000026_vis.jpg")

# 2. 读取图像 ---------------------------------------------------------------
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(img_path)
H, W = img.shape[:2]

# 3. 读取 YOLO 标签 ---------------------------------------------------------
with open(txt_path, 'r') as f:
    lines = [x.strip() for x in f if x.strip()]

# 4. 画框 ------------------------------------------------------------------
for line in lines:
    parts = line.split()
    cls = int(parts[0])
    xc, yc, w, h = map(float, parts[1:5])

    # 反归一化 → 像素坐标
    x1 = int((xc - w / 2) * W)
    y1 = int((yc - h / 2) * H)
    x2 = int((xc + w / 2) * W)
    y2 = int((yc + h / 2) * H)

    # 画框 + 类别文字
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, str(cls), (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# 5. 保存结果 ---------------------------------------------------------------
cv2.imwrite(save_path, img)
print("可视化结果已保存到：", save_path)