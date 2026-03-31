import cv2
import os
from ultralytics import YOLO

# ================== 配置区 ==================
MODEL_PATH = r"E:\final design\runs_smoke_3\yolov8_smoke_v3\weights\best.pt"
IMAGE_DIR = r"E:\final design\dataset\images\test"
CONF_THRES = 0.25
IMG_SIZE = (960, 640)
# ===========================================

def main():
    model = YOLO(MODEL_PATH)

    images = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ])

    if len(images) == 0:
        print("❌ test_smoke_image 里没有图片")
        return

    print(f"✅ 共检测 {len(images)} 张图片")
    print("操作说明：n 下一张 | q / ESC 退出")

    for name in images:
        img_path = os.path.join(IMAGE_DIR, name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        results = model(img, conf=CONF_THRES, verbose=False)
        r = results[0]

        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])

                cv2.rectangle(img, (x1, y1), (x2, y2),
                              (0, 0, 255), 2)
                cv2.putText(
                    img,
                    f"smoke {conf:.2f}",
                    (x1, max(y1 - 5, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

        show = cv2.resize(img, IMG_SIZE)
        cv2.imshow("Smoke Detection", show)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
