import cv2


def spatial_filter(img, smoke_results, person_results):
    """
    分层空间语义约束：
    1️⃣ 上半身硬约束
    2️⃣ 嘴部区域软权重
    3️⃣ 横向归一化限制
    """

    valid_smokes = []

    if len(smoke_results[0].boxes) == 0:
        return valid_smokes

    if len(person_results[0].boxes) == 0:
        return valid_smokes

    for s_box in smoke_results[0].boxes:

        sx1, sy1, sx2, sy2 = s_box.xyxy[0].cpu().numpy().astype(int)
        s_conf = float(s_box.conf[0])

        s_cx = (sx1 + sx2) // 2
        s_cy = (sy1 + sy2) // 2

        best_weight = 0

        for p_box in person_results[0].boxes:

            px1, py1, px2, py2 = p_box.xyxy[0].cpu().numpy().astype(int)

            person_height = py2 - py1
            person_width = px2 - px1

            # ==========================
            # ① 上半身硬约束
            # ==========================
            upper_limit = py1 + 0.6 * person_height
            if s_cy > upper_limit:
                continue

            # ==========================
            # ② 横向归一化限制
            # ==========================
            person_center_x = (px1 + px2) // 2
            if abs(s_cx - person_center_x) > 0.5 * person_width:
                continue

            # ==========================
            # ③ 嘴部软权重
            # ==========================
            mouth_top = py1 + 0.25 * person_height
            mouth_bottom = py1 + 0.45 * person_height

            if mouth_top <= s_cy <= mouth_bottom:
                location_weight = 1.0   # 嘴部区域
            else:
                location_weight = 0.6   # 上半身但非嘴部

            best_weight = max(best_weight, location_weight)

        if best_weight > 0:
            valid_smokes.append(
                (sx1, sy1, sx2, sy2, s_conf, best_weight)
            )

    return valid_smokes
