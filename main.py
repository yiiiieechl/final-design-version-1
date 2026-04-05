import cv2
import config
from detector import Detector
from spatial import spatial_filter
from temporal import TemporalJudge
from preprocess import preprocess_frame
import numpy as np
from risk import calculate_risk, risk_level

def main():

    detector = Detector(
        config.SMOKE_MODEL_PATH,
        config.POSE_MODEL_PATH
    )

    temporal = TemporalJudge(
        config.WINDOW_SIZE,
        config.TRIGGER_THRESHOLD
    )

    cap = cv2.VideoCapture(config.VIDEO_PATH)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    paused = False

    # 极简跳帧配置
    frame_skip = 10
    count = 0

    while True:

        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            current_frame += 1

        frame = preprocess_frame(frame)
        count += 1

        # ======================================
        # 【终极极简跳帧：第一帧强制检测，绝不报错】
        # ======================================
        if count % frame_skip == 0 or count == 1:
            # 你原版的检测代码（完全不动）
            smoke_results, person_results, pose_results = detector.detect(
                frame,
                config.SMOKE_CONF,
                config.PERSON_CONF
            )
            # 你原版的关键点绘制（完全不动）
            if pose_results[0].keypoints is not None:
                for kpts in pose_results[0].keypoints:
                    kpts_data = kpts.data.cpu().numpy()[0]
                    for i in range(5):
                        x, y, conf = kpts_data[i]
                        if conf > 0.5:
                            cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 255), -1)
            # 你原版的空间过滤（完全不动）
            valid_smokes = spatial_filter(frame, smoke_results, person_results, pose_results)

        # ======================================
        # 以下是你所有的原版代码，一行没改！
        # ======================================

        # 3️⃣ 画person框
        if person_results[0].boxes is not None:
            for box in person_results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)

        # 4️⃣ 空间过滤
        valid_smokes = spatial_filter(
            frame,
            smoke_results,
            person_results,
            pose_results
        )

        detected = len(valid_smokes) > 0
        temporal.update(detected)

        # 5️⃣ 风险计算
        ratio = temporal.get_ratio()
        risk_score = 0
        level, risk_color = "LOW", (0, 255, 0)

        if len(valid_smokes) > 0:
            max_conf = max([s[4] for s in valid_smokes])
            location_weight = max([s[5] for s in valid_smokes])
            risk_score = calculate_risk(max_conf, ratio, location_weight)
            level, risk_color = risk_level(risk_score)

        # 6️⃣ 画烟框
        for sx1, sy1, sx2, sy2, s_conf, location_weight in valid_smokes:
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2),
                          (0, 0, 255), 2)
            cv2.putText(frame,
                        f"SMOKE {s_conf:.2f}",
                        (sx1, sy1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2)

        # UI
        display_width = 960
        scale = display_width / frame.shape[1]
        display_height = int(frame.shape[0] * scale)
        frame = cv2.resize(frame, (display_width, display_height))

        h, w, _ = frame.shape
        panel_width = 280
        top_height = 50
        bottom_height = 30

        canvas = np.zeros((h + top_height + bottom_height, w + panel_width, 3), dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (w + panel_width, top_height), (30,30,30), -1)
        cv2.putText(canvas, "Construction Smoking Monitor", (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        canvas[top_height:top_height+h, 0:w] = frame

        panel_x = w + 15
        panel_y = top_height + 40
        cv2.putText(canvas, f"Frame: {current_frame}/{total_frames}", (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        panel_y +=30
        cv2.putText(canvas, f"Risk: {risk_score:.2f}", (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, risk_color, 2)
        panel_y +=30
        cv2.putText(canvas, f"Level: {level}", (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, risk_color, 3)

        y_base = top_height + h
        cv2.rectangle(canvas, (0, y_base), (w + panel_width, y_base + bottom_height), (40,40,40), -1)
        cv2.putText(canvas, "SPACE: Pause | A: Back | D: Forward | Q: Quit", (20, y_base+22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)

        cv2.imshow("Smoking Monitor", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord(' '): paused = not paused
        elif key == ord('d'):
            current_frame = min(current_frame+30, total_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        elif key == ord('a'):
            current_frame = max(current_frame-30, 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()