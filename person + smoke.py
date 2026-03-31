import cv2
from ultralytics import YOLO

# ===== 模型路径 =====
smoke_model = YOLO(r"E:\final design\runs_smoke_3\yolov8_smoke_v3\weights\best.pt")
person_model = YOLO("yolov8s.pt")

# ===== 测试图片 =====
img_path = r"E:\final design\dataset\images\test\abc193.jpg"

img = cv2.imread(img_path)

# ===== 推理 =====
smoke_results = smoke_model(img, conf=0.15, verbose=False)
person_results = person_model(img, conf=0.3, classes=[0], verbose=False)
# classes=[0] 只检测 person

# ===== 画 person 框（绿色）=====
for box in person_results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, "person", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# ===== 画 smoke 框（红色）=====
# ===== 空间约束判断 =====
for s_box in smoke_results[0].boxes:
    sx1, sy1, sx2, sy2 = s_box.xyxy[0].cpu().numpy().astype(int)
    s_conf = float(s_box.conf[0])

    # 计算 smoke 中心点
    s_cx = (sx1 + sx2) // 2
    s_cy = (sy1 + sy2) // 2

    smoke_inside_person = False

    # 遍历所有 person 框
    for p_box in person_results[0].boxes:
        px1, py1, px2, py2 = p_box.xyxy[0].cpu().numpy().astype(int)

        # 判断中心点是否在 person 框内
        if px1 <= s_cx <= px2 and py1 <= s_cy <= py2:
            smoke_inside_person = True
            break

    # ===== 只有在 person 内才画红框 =====
    if smoke_inside_person:
        cv2.rectangle(img, (sx1, sy1), (sx2, sy2), (0, 0, 255), 2)
        cv2.putText(img, f"SMOKING {s_conf:.2f}", (sx1, sy1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


# ===== 显示 =====
cv2.imshow("Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
