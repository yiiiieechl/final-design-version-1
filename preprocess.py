# preprocess.py
'''
import cv2

def preprocess_frame(frame):

    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return frame
'''
# 优化后的preprocess.py
import cv2
import numpy as np

def preprocess_frame(frame):
    # 1. 亮度自适应伽马校正（应对过暗/过曝）
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    gamma = 1.5 if mean_brightness < 80 else (0.7 if mean_brightness > 200 else 1.0)
    inv_gamma = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    frame = cv2.LUT(frame, lut)
    
    # 2. 双边滤波去噪（保留边缘，应对扬尘噪点）
    frame = cv2.bilateralFilter(frame, 9, 75, 75)
    
    # 3. 保留原CLAHE增强（LAB空间）
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # 4. 小目标增强（可选：超分辨率）
    # sr = cv2.dnn_superres.DnnSuperResImpl_create()
    # sr.readModel("ESPCN_x4.pb")  # 需下载预训练模型
    # sr.setModel("espcn", 4)
    # frame = sr.upsample(frame)
    
    return frame
