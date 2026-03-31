import cv2
import os

# 配置路径
img_dir = "E:/final design/dataset/images/train"
label_dir = "E:/final design/dataset/labels/train"
classes = ["smoke", "person"]  # 和classes.txt一致

# 读取一张标注后的图片和标签
img_name = "2.jpg"  # 替换成你标注的图片名
img_path = os.path.join(img_dir, img_name)
label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt"))  # 适配png的话改成.replace(".png", ".txt")

# 读取图片
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2] # 获取图片的高和宽

# 读取标注文件并画框
with open(label_path, "r") as f:
    for line in f.readlines(): # 每一行代表一个目标
        cls_id, x_center, y_center, width, height = map(float, line.strip().split()) # 转成浮点数
        # 归一化坐标转像素坐标
        x1 = int((x_center - width/2) * w)
        y1 = int((y_center - height/2) * h)
        x2 = int((x_center + width/2) * w)
        y2 = int((y_center + height/2) * h)
        # 画框+标注类别
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 蓝色框
        cv2.putText(img_rgb, classes[int(cls_id)], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# 显示标注后的图片
import matplotlib.pyplot as plt
plt.imshow(img_rgb)
plt.axis("off")
plt.title("标注效果验证")
plt.show()
print("✅ 标注验证完成！框体和目标匹配即说明格式正确。")