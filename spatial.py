import cv2
import numpy as np

def spatial_filter(img, smoke_results, person_results, pose_results=None):
    """
    基于人体关键点的空间过滤
    """
    valid_smokes = []
    
    # 基础检查
    if len(smoke_results[0].boxes) == 0:
        return valid_smokes
    if len(person_results[0].boxes) == 0:
        return valid_smokes

    # 遍历每一个检测到的烟雾
    for s_box in smoke_results[0].boxes:
        sx1, sy1, sx2, sy2 = s_box.xyxy[0].cpu().numpy().astype(int)
        s_conf = float(s_box.conf[0])
        s_cx = (sx1 + sx2) // 2  # 烟雾中心点X
        s_cy = (sy1 + sy2) // 2  # 烟雾中心点Y
        
        best_weight = 0

        # 遍历每一个检测到的人 (尝试匹配)
        # 注意：这里为了简单，我们假设 person_results 和 pose_results 的顺序是一致的
        # 如果要做严格的ID匹配，需要用到 ByteTrack 等多目标跟踪，这里先用简化版
        for p_idx, p_box in enumerate(person_results[0].boxes):
            px1, py1, px2, py2 = p_box.xyxy[0].cpu().numpy().astype(int)
            person_height = py2 - py1
            person_width = px2 - px1
            
            # ==========================================
            # 1. 上半身硬约束 (必须满足)
            # ==========================================
            # 烟雾中心点必须在人体上半身 (从上往下60%)
            upper_body_limit = py1 + 0.6 * person_height
            if s_cy > upper_body_limit:
                continue
            
            # ==========================================
            # 2. 横向距离硬约束 (必须满足)
            # ==========================================
            # 烟雾中心点不能离人体中心太远
            person_center_x = (px1 + px2) // 2
            if abs(s_cx - person_center_x) > 0.6 * person_width: # 稍微放宽一点到0.6
                continue

            # ==========================================
            # 3. 关键点软权重 (核心改进)
            # ==========================================
            current_weight = 0.5 # 基础保底权重
            
            # 检查是否有Pose结果，且当前索引有人
            if pose_results is not None and len(pose_results[0]) > p_idx:
                # 获取关键点数据 (shape: [17, 3] -> [x, y, conf])
                # 0:鼻子, 1:左眼, 2:右眼, 3:左耳, 4:右耳...
                kpts = pose_results[0][p_idx].keypoints.data.cpu().numpy()[0]
                
                # 提取脸部关键点 (鼻子、双眼)
                # 我们只取置信度 > 0.5 的关键点
                face_points_y = []
                
                # 鼻子 (索引 0)
                if kpts[0][2] > 0.5: 
                    face_points_y.append(kpts[0][1])
                # 左眼 (索引 1)
                if kpts[1][2] > 0.5:
                    face_points_y.append(kpts[1][1])
                # 右眼 (索引 2)
                if kpts[2][2] > 0.5:
                    face_points_y.append(kpts[2][1])

                # 如果检测到了可靠的脸部关键点
                if len(face_points_y) > 0:
                    # 计算脸部平均Y坐标
                    face_avg_y = np.mean(face_points_y)
                    
                    # 假设嘴部在脸部下方一点的位置
                    # 这是一个经验区域：脸部平均Y 到 脸部平均Y + 0.25*身高
                    mouth_zone_top = face_avg_y
                    mouth_zone_bottom = face_avg_y + 0.25 * person_height
                    
                    # 判断烟雾是否在这个黄金区域
                    if mouth_zone_top <= s_cy <= mouth_zone_bottom:
                        current_weight = 1.0  # 嘴部区域，权重最高
                    else:
                        # 不在嘴部，但在上半身，权重稍低
                        current_weight = 0.7
                else:
                    # 关键点不可见时，回退到原始的固定比例法
                    mouth_top = py1 + 0.25 * person_height
                    mouth_bottom = py1 + 0.45 * person_height
                    if mouth_top <= s_cy <= mouth_bottom:
                        current_weight = 0.9 # 回退模式权重稍低
                    else:
                        current_weight = 0.6
            else:
                # 没有Pose模型结果时的回退逻辑
                current_weight = 0.6

            best_weight = max(best_weight, current_weight)

        if best_weight > 0:
            valid_smokes.append(
                (sx1, sy1, sx2, sy2, s_conf, best_weight)
            )
            
    return valid_smokes