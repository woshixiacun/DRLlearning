import sys
sys.path.remove('/mnt/d/Study_File/codezcy/DRLlearning/ultralytics-main')
from ultralytics import YOLO
# 加载一个预训练的 YOLO11n 模型
# model = YOLO("yolo11n.pt")
model = YOLO("/mnt/d/Study_File/codezcy/DRLlearning/weights/yolo11n.pt")

# 在 COCO8 数据集上训练模型 100 个周期
train_results = model.train(
    data="/mnt/d/Study_File/codezcy/DRLlearning/ultralytics-main/ultralytics/cfg/datasets/mydatya/coco8.yaml",  # 数据集配置文件路径
    epochs=10,  # 训练周期数
    imgsz=640,  # 训练图 像尺寸
    device="cuda",  # 运行设备（例如 'cpu', 0, [0,1,2,3]）
)

# 评估模型在验证集上的性能
metrics = model.val()

# 对图像执行目标检测
import cv2

results = model("/mnt/d/Study_File/codezcy/DRLlearning/ultralytics-main/datasets/coco8/images/val/000000000036.jpg")
im_array = results[0].plot()  # 得到带框的 numpy 数组 (BGR 格式)
cv2.imwrite("/mnt/d/Study_File/codezcy/DRLlearning/ultralytics-main/pred_output.jpg", im_array)

# 将模型导出为 ONNX 格式以进行部署
# path = model.export(format="onnx")  # 返回导出模型的路径