import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.BaseDetector import baseDet
from utils.torch_utils import select_device
from utils.datasets import letterbox

class Detector(baseDet):

    def __init__(self):
        super(Detector, self).__init__()
        self.init_model()
        self.build_config()

    def init_model(self):

        self.weights = 'weights/yolov5s.pt'
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(self.weights, map_location=self.device)
        model.to(self.device).eval()
        model.half()
        # torch.save(model, 'test.pt')
        self.m = model
        self.names = model.module.names if hasattr(
            model, 'module') else model.names

    def preprocess(self, img):

        img0 = img.copy()  # 500 888 3
        img = letterbox(img, new_shape=self.img_size)[0] #384 640 3
        img = img[:, :, ::-1].transpose(2, 0, 1) # 3 384 640   第一步BGR → RGB。OpenCV 读出来是 BGR 顺序，PyTorch 预训练模型习惯 RGB，于是用切片倒序把通道换过来。第二步HWC → CHW。
        img = np.ascontiguousarray(img)  # 把数组变成内存连续。
        img = torch.from_numpy(img).to(self.device)  #numpy → tensor，并搬到 GPU/CPU。
        img = img.half()  # 半精度 FP32 → FP16（半精度）。
        img /= 255.0  # 图像归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  #补 batch 维度。现在 tensor 形状是 (C,H,W)，模型要求 (N,C,H,W)。unsqueeze(0) 在最前面加一维，变成 1 张图的 batch。

        return img0, img

    def detect(self, im):

        im0, img = self.preprocess(im)

        pred = self.m(img, augment=False)[0]
        pred = pred.float()
        pred = non_max_suppression(pred, self.threshold, 0.4)

        pred_boxes = []
        for det in pred:

            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]
                    if not lbl in ['person', 'car', 'truck']:
                        continue
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    pred_boxes.append(
                        (x1, y1, x2, y2, lbl, conf))

        return im, pred_boxes

