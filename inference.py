import torch
from PIL import Image
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

def _best_mask_by_area(gray):
    # 尝试两种阈值方向，选出最大连通域更大的那个
    masks = []
    for th in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, m = cv2.threshold(gray, 0, 255, th + cv2.THRESH_OTSU)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
        masks.append(m)
    # 选择最大轮廓面积更大的mask
    def largest_area(mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: 
            return 0, None
        c = max(cnts, key=cv2.contourArea)
        return cv2.contourArea(c), c
    areas = [largest_area(m)[0] for m in masks]
    idx = int(np.argmax(areas))
    cnt = largest_area(masks[idx])[1]
    return masks[idx], cnt

def get_rotated_rect_from_box(img_bgr, box):
    x1, y1, x2, y2 = map(int, box[:4])
    x1, y1 = max(x1,0), max(y1,0)
    x2, y2 = min(x2, img_bgr.shape[1]-1), min(y2, img_bgr.shape[0]-1)
    if x2 <= x1 or y2 <= y1:
        return None, None

    roi = img_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None, None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 选择更能代表物体的mask与最大轮廓
    mask, cnt = _best_mask_by_area(gray)
    if cnt is None or cv2.contourArea(cnt) < 100:  # 过滤过小区域
        # 退化为水平框
        return None, None

    rect = cv2.minAreaRect(cnt)               # ((cx,cy),(w,h),angle) in ROI
    box_pts = cv2.boxPoints(rect)             # 4x2
    box_pts += np.array([x1, y1])             # 偏移回原图
    box_pts = box_pts.astype(int)
    # 将 rect 的中心坐标也偏移回原图
    (cx, cy), (w, h), angle = rect
    rect_global = ((cx + x1, cy + y1), (w, h), angle)
    return rect_global, box_pts

# 在原图上绘制旋转矩形（红色），并标注角度与类别名
rotated_draw = img_cv.copy()
for box in boxes:
    rect, pts = get_rotated_rect_from_box(img_cv, box)
    if pts is not None:
        cv2.drawContours(rotated_draw, [pts], -1, (0, 0, 255), 2)
        # 标注类别与角度
        angle = rect[2]
        cx, cy = map(int, rect[0])
        cv2.putText(rotated_draw, f"陶钢过滤网 {angle:.1f} deg", (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)

# 保存两份结果：1) 仅YOLO水平框；2) 含旋转框
os.makedirs('results_new', exist_ok=True)
yolo_out = os.path.join('results_new', 'yolo_result.jpg')
rot_out  = os.path.join('results_new', 'rotated_result.jpg')

# 这两行只会保存YOLO的水平框（用于对比）
results.save(save_dir='results_new/')
# 保存旋转框结果（你要看的斜框在这张图里）
cv2.imwrite(rot_out, rotated_draw)
print(f"已保存：YOLO水平框目录 = {yolo_out}（文件名由YOLO生成）")
print(f"已保存：旋转框结果 = {rot_out}")
