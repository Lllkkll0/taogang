import cv2
import numpy as np

# 读取图像
img = cv2.imread("your_image.jpg", 0)  # 读取为灰度图
if img is not None:
    t1, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    print(t1)  # 输出阈值
else:
    print("图像读取失败")