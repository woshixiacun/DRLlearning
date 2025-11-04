import sys
# sys.path.remove('C:/Users/Clavi/Desktop/coding/DRLlearning/ultralytics-main')
from ultralytics import YOLO
from ultralytics import YOLO

if __name__ == "__main__":

    # 用自定义结构初始化模型
    model = YOLO("C:/Users/Clavi/Desktop/coding/DRLlearning/ultralytics-main/ultralytics/cfg/models/v8/yolov8-zcy.yaml")
    # # 加载一个预训练的 YOLO8n 模型。 新版 Ultralytics YOLO（v8.2+）会自动尝试加载权重（内部已经智能匹配部分层）。
    model.load("C:/Users/Clavi/Desktop/coding/DRLlearning/weights/yolov8n.pt")

    # 在 COCO8 数据集上训练模型 100 个周期
    train_results = model.train(
        data= "C:/Users/Clavi/Desktop/coding/DRLlearning/ultralytics-main/ultralytics/cfg/datasets/mydatya/neudet.yaml",
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

    results = model("C:/Users/Clavi/Desktop/coding/DRLlearning/ultralytics-main/datasets/NEU_DET/mydata/images/train/inclusion_285.jpg")
    im_array = results[0].plot()  # 得到带框的 numpy 数组 (BGR 格式)
    cv2.imwrite("C:/Users/Clavi/Desktop/coding/DRLlearning/ultralytics-main/pred_crazing_1.jpg", im_array)

    # 将模型导出为 ONNX 格式以进行部署
    # path = model.export(format="onnx")  # 返回导出模型的路径