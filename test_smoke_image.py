import cv2
from ultralytics import YOLO

# ========== 1. 加载模型 ==========
model = YOLO(r"E:\final design\runs_smoke_2\yolov8_smoke_v2\weights\best.pt")

# ========== 2. 读入图片（一定要是你标过的） ==========
img_path = r"E:\final design\dataset\images\train\smoking_0002.jpg"
img = cv2.imread(img_path)

assert img is not None, "❌ 图片没读进来，检查路径"

# ========== 3. 推理（极低阈值，调试用） ==========
results = model(
    img,
    conf=0.05,      # 🔴 极低阈值
    iou=0.3,
    verbose=True   # 🔴 打印模型内部信息
)

r = results[0]

print(f"\n👉 检测到的 box 数量: {len(r.boxes) if r.boxes is not None else 0}")

# ========== 4. 无论如何都画 ==========
if r.boxes is not None and len(r.boxes) > 0:
    for i, box in enumerate(r.boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        print(f"Box {i}: conf={conf:.3f}, cls={cls}")

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            img,
            f"smoke {conf:.2f}",
            (x1, max(y1 - 5, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )
else:
    print("⚠️ 模型没有输出任何框（即使在低 conf 下）")

# ========== 5. 显示 ==========
cv2.imshow("YOLOv8 Smoke Debug", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
