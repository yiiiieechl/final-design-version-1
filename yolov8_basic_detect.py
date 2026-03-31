import  cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np


IMG_PATH = "C:/Users/Leon Chan/Desktop/test.jpg"

MODEL_PATH = "yolov8n.pt"

model = YOLO(MODEL_PATH)
print("yolo模型加载成功")

img = cv2.imread(IMG_PATH)

if img is None:
    raise ValueError("图片读取失败")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(f"图片大小为：{img.shape}")

results = model(img_rgb, device=0)
print("✅ 检测完成！")

result = results[0]
annotated_img = result.plot()

plt.figure(figsize=(12, 8))  # 设置图片显示大小（12×8英寸）
plt.imshow(annotated_img)     # 显示图片
plt.title("YOLOv8n Detection Result (Construction Site)")  # 图片标题
plt.axis("off")              # 隐藏坐标轴（更美观）
plt.show()                   # 弹出显示窗口

# 保存检测后的图片到项目文件夹（转换回BGR格式，OpenCV保存需要）
cv2.imwrite("detected_site.jpg", cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
print("✅ 检测结果已保存为detected_site.jpg！")



print("\n 检测到的目标列表：")
# 遍历所有检测框，打印类别和置信度
for box in result.boxes:
    cls_id = int(box.cls[0])        # 类别ID（比如0=人，2=车）
    cls_name = result.names[cls_id] # 类别名称（比如person、car）
    confidence = float(box.conf[0]) # 置信度（0-1，越高越准确）
    print(f"- 类别：{cls_name} | 置信度：{confidence:.2f}")
