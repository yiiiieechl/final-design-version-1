from ultralytics import YOLO
import torch

class Detector:
    def __init__(self, smoke_model_path, pose_model_path):
        # 强制GPU，无损耗
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"✅ 运行设备: {self.device}")

        # 加载你的高精度YOLOv8m香烟模型 + 轻量姿态模型
        self.smoke_model = YOLO(smoke_model_path).to(self.device)
        self.pose_model = YOLO(pose_model_path).to(self.device)

        # 🔥 模型融合：唯一【无损提速】方法（速度变快，精度不变）
        self.smoke_model.fuse()
        self.pose_model.fuse()

    def detect(self, img, smoke_conf, person_conf):
        smoke_results = self.smoke_model(
        img, 
        conf=smoke_conf, 
        device=self.device, 
        verbose=False,
        imgsz=480,
        half=True,  # 添加：FP16半精度推理
        agnostic_nms=False
    )
        pose_results = self.pose_model(
        img, 
        conf=person_conf, 
        device=self.device, 
        verbose=False,
        imgsz=640,
        half=True   # 添加：FP16半精度推理
    )
        return smoke_results, pose_results, pose_results