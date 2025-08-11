import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os

print("正在加载新训练的模型...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/best.pt')
model.names[0] = '陶钢过滤网'
model.conf = 0.25
model.iou = 0.45

# 读取图片
img_path = r'sample_data/images/train/40.jpg'
print(f"正在读取图片: {img_path}")
img_pil = Image.open(img_path)
img = np.array(img_pil)
img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# 推理（获得水平框）
print("正在进行推理...")
results = model(img)
boxes = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, cls]

# 只处理检测到的第一个过滤网区域
if len(boxes) > 0:
    x1, y1, x2, y2 = map(int, boxes[0][:4])
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, img_cv.shape[1]-1), min(y2, img_cv.shape[0]-1)
    roi = img_cv[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 直方图均衡化增强对比度
    gray_eq = cv2.equalizeHist(gray)
    # 保存均衡化结果
    os.makedirs('results_new', exist_ok=True)
    cv2.imwrite('results_new/roi_gray_eq.jpg', gray_eq)
    # 自适应阈值分割
    binary = cv2.adaptiveThreshold(gray_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 5)
    # 形态学操作去除小噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    # 保存二值化和形态学处理结果
    cv2.imwrite('results_new/roi_binary.jpg', binary)
    cv2.imwrite('results_new/roi_clean.jpg', clean)
    # 查找轮廓
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 过滤面积过小的孔洞
    min_hole_area = 30  # 可根据实际调整
    holes = [cnt for cnt in contours if cv2.contourArea(cnt) > min_hole_area]
    hole_count = len(holes)
    # 在ROI上画出孔洞
    roi_draw = roi.copy()
    cv2.drawContours(roi_draw, holes, -1, (0, 0, 255), 2)
    # 在原图上合成ROI
    result_img = img_cv.copy()
    result_img[y1:y2, x1:x2] = roi_draw
    # 用Pillow写中文，左上角显示
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(result_img_rgb)
    draw = ImageDraw.Draw(result_pil)
    font_path = "C:/Windows/Fonts/simhei.ttf"
    if not os.path.exists(font_path):
        font_path = None
    try:
        font = ImageFont.truetype(font_path, 36) if font_path else ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    draw.text((20, 20), f"孔洞数: {hole_count}", font=font, fill=(255,0,0))
    # 保存结果
    out_path = os.path.join('results_new', 'hole_result.jpg')
    result_pil.save(out_path)
    print(f"已检测到孔洞数: {hole_count}，结果已保存到 {out_path}")
    print("中间处理结果已保存到 results_new/ 目录，便于调参和观察效果。")
else:
    print("未检测到陶钢过滤网区域，无法统计孔洞数。")
