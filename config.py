# ========================
# 训练参数配置文件
# ========================

# 模型
MODEL_NAME = "yolov8m.pt"

# 数据
DATA_PATH = r"E:\final design\dataset\data.yaml"

# 训练参数
EPOCHS = 200
IMG_SIZE = 640
BATCH_SIZE = 4
DEVICE = 0
WORKERS = 4

# 数据增强
MOSAIC = 0.5
MIXUP = 0.0
CLOSE_MOSAIC = 10
SCALE = 0.5

HSV_H = 0.015
HSV_S = 0.5
HSV_V = 0.3

FLIPUD = 0.0
FLIPLR = 0.5

# 输出路径
PROJECT_NAME = "runs_smoke_3"
RUN_NAME = "yolov8_smoke_v3"
EXIST_OK = True


# ========================
# 模型路径
# ========================
SMOKE_MODEL_PATH = r"E:\final design\runs_smoke_3\yolov8_smoke_v3\weights\best.pt"
PERSON_MODEL_PATH = "yolov8s.pt"

# ========================
# 推理参数
# ========================
SMOKE_CONF = 0.15
PERSON_CONF = 0.3

# ========================
# 测试图片
# ========================
IMAGE_PATH = r"E:\final design\dataset\images\test\abc193.jpg"


# ========================
# 视频路径
# ========================
VIDEO_PATH = r"E:\final design\video\smoking2.mp4"

# ========================
# 时间窗口参数
# ========================
WINDOW_SIZE = 30
TRIGGER_THRESHOLD = 5
RISK_THRESHOLD = 0.6
