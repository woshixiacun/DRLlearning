import sys
sys.path.remove('/mnt/d/Study_File/codezcy/DRLlearning/ultralytics-main')
from ultralytics import YOLO

# 用自定义结构初始化模型
model = YOLO("/mnt/d/Study_File/codezcy/DRLlearning/ultralytics-main/ultralytics/models/v8/yolov8-custom.yaml")

# 加载一个预训练的 YOLO8n 模型
model = YOLO("/mnt/d/Study_File/codezcy/DRLlearning/weights/yolov8n.pt")
# model = YOLO("/mnt/d/Study_File/codezcy/DRLlearning/runs/detect/train/weights/best.pt")


# 在 COCO8 数据集上训练模型 100 个周期
train_results = model.train(
    # data="/mnt/d/Study_File/codezcy/DRLlearning/ultralytics-main/ultralytics/cfg/datasets/mydatyasets/coco8.yaml",  # 数据集配置文件路径
    # imgsz=640,  # 训练图 像尺寸
    data= "/mnt/d/Study_File/codezcy/DRLlearning/ultralytics-main/ultralytics/cfg/datasets/mydatya/neudet.yaml",
    imgsz=200,
    batch=128,
    epochs=100,  # 训练周期数
    multi_scale=True, # 多尺度
    cos_lr=True,  # 使用余弦学习率调度器
    optimizer="AdamW",
    # warmup_epochs=10,
    classes=[1,2,3,4,5],
    device="cuda",  # 运行设备（例如 'cpu', 0, [0,1,2,3]）
)

# 评估模型在验证集上的性能
metrics = model.val()

# 对图像执行目标检测
import cv2

# results = model("/mnt/d/Study_File/codezcy/DRLlearning/ultralytics-main/datasets/coco8/images/val/000000000036.jpg")
results = model("/mnt/d/Study_File/codezcy/DRLlearning/ultralytics-main/datasets/NEU_DET/mydata/images/train/inclusion_285.jpg")
im_array = results[0].plot()  # 得到带框的 numpy 数组 (BGR 格式)
cv2.imwrite("/mnt/d/Study_File/codezcy/DRLlearning/ultralytics-main/pred_crazing_1.jpg", im_array)

# 将模型导出为 ONNX 格式以进行部署
# path = model.export(format="onnx")  # 返回导出模型的路径