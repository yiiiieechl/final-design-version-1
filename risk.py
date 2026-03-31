def calculate_risk(confidence, temporal_ratio, location_weight=1.0):
    """
    confidence: YOLO置信度
    temporal_ratio: 时间窗口内检测比例
    location_weight: 区域权重（暂时设1）

    返回 0~1 之间的风险分数
    """

    alpha = 0.4
    beta = 0.4
    gamma = 0.2

    score = alpha * confidence + beta * temporal_ratio + gamma * location_weight

    return min(score, 1.0)


def risk_level(score):
    """
    根据风险值划分等级
    """

    if score >= 0.7:
        return "HIGH", (0, 0, 255)
    elif score >= 0.4:
        return "MEDIUM", (0, 165, 255)
    else:
        return "LOW", (0, 255, 0)
