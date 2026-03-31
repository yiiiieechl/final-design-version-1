import cv2
from ultralytics import YOLO
from collections import deque

# ===== 模型加载 =====
smoke_model = YOLO(r"E:\final design\runs_smoke_3\yolov8_smoke_v3\weights\best.pt")
person_model = YOLO("yolov8s.pt")

# ===== 视频路径 =====
video_path = r"E:\final design\video\smoking2.mp4"
cap = cv2.VideoCapture(video_path)

# ===== 视频级判定参数 =====
window_size = 30
trigger_threshold = 5
smoking_window = deque(maxlen=window_size)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ===== 推理 =====
    smoke_results = smoke_model(frame, conf=0.15, verbose=False)
    person_results = person_model(frame, conf=0.3, classes=[0], verbose=False)

    frame_smoking = False  # ← 本帧是否检测到吸烟

    # ===== 画 person 框 =====
    for p_box in person_results[0].boxes:
        px1, py1, px2, py2 = p_box.xyxy[0].cpu().numpy().astype(int)
        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
        cv2.putText(frame, "person", (px1, py1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ===== 空间约束 + smoke =====
    for s_box in smoke_results[0].boxes:
        sx1, sy1, sx2, sy2 = s_box.xyxy[0].cpu().numpy().astype(int)
        s_conf = float(s_box.conf[0])

        s_cx = (sx1 + sx2) // 2
        s_cy = (sy1 + sy2) // 2

        smoke_inside_upper = False

        for p_box in person_results[0].boxes:
            px1, py1, px2, py2 = p_box.xyxy[0].cpu().numpy().astype(int)
            person_h = py2 - py1
            upper_y2 = py1 + int(person_h * 0.6)  # ← 建议放宽到 0.6

            if px1 <= s_cx <= px2 and py1 <= s_cy <= upper_y2:
                smoke_inside_upper = True
                break

        if smoke_inside_upper:
            frame_smoking = True  # ← 标记本帧检测成功

            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 0, 255), 2)
            cv2.putText(frame, f"SMOKING {s_conf:.2f}", (sx1, sy1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # ===== 滑动窗口统计 =====
    smoking_window.append(1 if frame_smoking else 0)

    if sum(smoking_window) >= trigger_threshold:
        cv2.putText(frame, "SMOKING DETECTED",
                    (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3)

    cv2.imshow("Video Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
