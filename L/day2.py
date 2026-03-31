import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

IMG_PATH_NORMAL = "C:/Users/Leon Chan/Desktop/picture/site_normal.jpg"  # 普通工地图路径
IMG_PATH_BACKLIGHT = "C:/Users/Leon Chan/Desktop/picture/site_backlight.png"

def read_and_convert_img(img_path):
    """
    读取图片并转化为RGB
    """
    if not os.path.exists(img_path):
         raise ValueError(f"{img_path} does not exist")

    img_bgr = cv2.imread(img_path)
    #转化格式
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f"图片读取成功尺寸{img_bgr.shape}")

    return img_rgb,img_bgr


def clahe_img(img_rgb):
    """
    对彩色图的亮度通道做CLAHE增强，保留色彩
    参数：img_rgb - 输入的RGB彩色图
    返回：增强后的RGB彩色图
    """
    # 1. 把RGB转成HSV（H:色调, S:饱和度, V:亮度）
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    # 2. 提取亮度通道（V通道）
    v_channel = img_hsv[:, :, 2]
    # 3. 对亮度通道做CLAHE增强
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
    v_channel_clahe = clahe.apply(v_channel)
    # 4. 把增强后的亮度通道放回HSV，再转回RGB
    img_hsv[:, :, 2] = v_channel_clahe
    img_clahe_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

    return img_clahe_rgb






# ---------------------- 执行函数：读取两张图片 ----------------------
# 普通图片
img_normal_bgr, img_normal_rgb = read_and_convert_img(IMG_PATH_NORMAL)
# 逆光图片
img_backlight_bgr, img_backlight_rgb = read_and_convert_img(IMG_PATH_BACKLIGHT)
# ---------------------- 操作1：图片缩放（统一YOLO输入尺寸640×640） ----------------------
img_normal_resize = cv2.resize(img_normal_rgb, (640,640))
img_backlight_resize = cv2.resize(img_backlight_rgb, (640,640))

print("图片尺寸放缩后",img_normal_resize.shape)
# ---------------------- 操作2：灰度转换（简化特征，减少计算） ----------------------
img_normal_gray = cv2.cvtColor(img_normal_rgb, cv2.COLOR_RGB2GRAY)
img_backlight_gray = cv2.cvtColor(img_backlight_rgb, cv2.COLOR_RGB2GRAY)
# ---------------------- 操作3：逆光增强（CLAHE，重点！） ----------------------
'''
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_backlight_gray_clahe = clahe.apply(img_backlight_gray)
# 转回RGB格式（方便显示）
img_backlight_clahe = cv2.cvtColor(img_backlight_gray_clahe, cv2.COLOR_GRAY2RGB)
print("✅ 逆光增强完成！")
'''
img_backlight_clahe = clahe_img(img_backlight_rgb)
# ---------------------- 可视化对比（保存毕设素材） ----------------------
plt.figure(figsize=(16,8))

# 1. 普通图片对比：原图 → 缩放 → 灰度
plt.subplot(2, 3, 1)
plt.imshow(img_normal_rgb)
plt.title("normal")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(img_normal_resize)
plt.title("resize640×640")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(img_normal_gray, cmap="gray")
plt.title("gray")
plt.axis("off")
# 2. 逆光图片对比：原图 → 灰度 → CLAHE增强
plt.subplot(2, 3, 4)
plt.imshow(img_backlight_rgb)
plt.title("backlight")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(img_backlight_gray, cmap="gray")
plt.title("backlight_gray")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(img_backlight_clahe)
plt.title("img_backlight_clahe")
plt.axis("off")

SAVE_PATH = "E:/毕业设计/preprocess_results/preprocess_result2.jpg"
plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
plt.show()
print("✅ 所有预处理完成！对比图已保存为preprocess_result.jpg")