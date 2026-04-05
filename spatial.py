import torch
import numpy as np

def spatial_filter(img, smoke_results, person_results, pose_results=None):
    """向量化空间过滤：零CUDA同步，批量处理"""
    valid_smokes = []
    
    if len(smoke_results[0].boxes) == 0 or len(person_results[0].boxes) == 0:
        return valid_smokes
    
    # ========== 批量提取张量（只同步1次） ==========
    device = smoke_results[0].boxes.xyxy.device
    
    # 烟雾数据 [N, 4]
    s_boxes = smoke_results[0].boxes.xyxy
    s_confs = smoke_results[0].boxes.conf
    n_smoke = len(s_boxes)
    
    # 人体数据 [M, 4]
    p_boxes = person_results[0].boxes.xyxy
    n_person = len(p_boxes)
    
    # 计算几何特征（GPU上）
    s_cx = (s_boxes[:, 0] + s_boxes[:, 2]) / 2
    s_cy = (s_boxes[:, 1] + s_boxes[:, 3]) / 2
    p_h = p_boxes[:, 3] - p_boxes[:, 1]
    p_w = p_boxes[:, 2] - p_boxes[:, 0]
    p_cx = (p_boxes[:, 0] + p_boxes[:, 2]) / 2
    p_top = p_boxes[:, 1]
    upper_limit = p_top + 0.6 * p_h
    
    # ========== 向量化硬约束 [N, M] ==========
    s_cx_exp = s_cx.unsqueeze(1)
    s_cy_exp = s_cy.unsqueeze(1)
    
    in_upper = s_cy_exp <= upper_limit.unsqueeze(0)
    in_width = torch.abs(s_cx_exp - p_cx.unsqueeze(0)) <= (0.6 * p_w).unsqueeze(0)
    valid_mask = in_upper & in_width
    
    if valid_mask.sum() == 0:
        return valid_smokes
    
    # ========== 关键点软权重（如有） ==========
    weights = torch.full((n_smoke, n_person), 0.6, device=device)
    
    if pose_results is not None and len(pose_results[0]) > 0:
        kpts = pose_results[0].keypoints.data  # [M, 17, 3]
        face = kpts[:, 0:3, :]  # 鼻子、眼、眼 [M, 3, 3]
        face_conf = face[:, :, 2]
        face_y = face[:, :, 1]
        
        valid_face = face_conf > 0.5
        face_count = valid_face.sum(dim=1)
        face_y_mean = (face_y * valid_face).sum(dim=1) / (face_count + 1e-6)
        
        mouth_top = face_y_mean
        mouth_bottom = face_y_mean + 0.25 * p_h
        
        in_mouth = (s_cy_exp >= mouth_top.unsqueeze(0)) & \
                   (s_cy_exp <= mouth_bottom.unsqueeze(0))
        
        has_face = (face_count > 0).unsqueeze(0)
        weights = torch.where(has_face & in_mouth, 1.0, 
                             torch.where(has_face, 0.7, 0.6))
    
    # ========== 筛选结果（最后才转CPU） ==========
    valid_pairs = valid_mask.nonzero(as_tuple=False)
    pair_weights = weights[valid_pairs[:, 0], valid_pairs[:, 1]]
    
    # 取每个烟雾的最大权重
    best_weights = {}
    for idx, (s_idx, _) in enumerate(valid_pairs.tolist()):
        w = float(pair_weights[idx])
        if s_idx not in best_weights or w > best_weights[s_idx]:
            best_weights[s_idx] = w
    
    # 组装结果（统一numpy转换）
    s_boxes_np = s_boxes.cpu().numpy()
    s_confs_np = s_confs.cpu().numpy()
    
    for s_idx, w in best_weights.items():
        if w > 0:
            x1, y1, x2, y2 = s_boxes_np[s_idx]
            valid_smokes.append((int(x1), int(y1), int(x2), int(y2), 
                               float(s_confs_np[s_idx]), w))
    
    return valid_smokes