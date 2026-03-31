import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8m-pose.pt")
cap = cv2.VideoCapture("smoking.mp4")

POSE_THRESH = 70  # 像素阈值
WINDOW_SIZE = 30  # N 帧
POSE_COUNT = 20   # M 帧

pose_history = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    r = results[0]

    pose_ok = False  # 每帧重置

    if r.keypoints is not None:
        keypoints = r.keypoints.xy.cpu().numpy()  # (N,17,2)

        for person_kps in keypoints:
            if person_kps.shape[0] < 11:
                continue

            nose = person_kps[0]
            left_wrist = person_kps[9]
            right_wrist = person_kps[10]

            dist_left = np.linalg.norm(nose - left_wrist)
            dist_right = np.linalg.norm(nose - right_wrist)

            # 画关键点
            cv2.circle(frame, tuple(nose.astype(int)), 5, (0, 0, 255), -1)
            cv2.circle(frame, tuple(left_wrist.astype(int)), 5, (0, 255, 0), -1)
            cv2.circle(frame, tuple(right_wrist.astype(int)), 5, (255, 0, 0), -1)

            # 姿态判别
            if min(dist_left, dist_right) < POSE_THRESH:
                pose_ok = True
            # ===== 连续帧统计 =====
            pose_history.append(1 if pose_ok else 0)

            if len(pose_history) > WINDOW_SIZE:
                pose_history.pop(0)

            continuous_pose_ok = sum(pose_history) >= POSE_COUNT

    # === 在 for 外统一打字 ===
    if continuous_pose_ok:
        cv2.putText(frame, "smoking pose",
                    (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 0, 255), 3)

    show = cv2.resize(frame, (960, 700))
    cv2.imshow("pose", show)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
