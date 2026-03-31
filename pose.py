import cv2
from ultralytics import YOLO

# 1. 加载姿态模型（第一次会自动下载）
model = YOLO("yolov8m-pose.pt")

# 2. 打开视频（0 = 摄像头，也可以换成视频文件）
cap = cv2.VideoCapture("xuanbu.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 3. 姿态检测
    results = model(frame)

    # 4. 遍历检测结果
    for r in results:
        if r.keypoints is None:
            continue

        keypoints = r.keypoints.xy.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()

        for i, kps in enumerate(keypoints):
            x1, y1, x2, y2 = boxes[i].astype(int)

            # 画人框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 画关键点
            for kp in kps:
                x, y = int(kp[0]), int(kp[1])
                if x == 0 and y == 0:
                    continue  # 关键点缺失
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    cv2.imshow("YOLOv8 Pose", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
