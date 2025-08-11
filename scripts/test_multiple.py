import torch
from PIL import Image
import os
from datetime import datetime

# 创建带时间戳的结果文件夹
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_folder = rf'G:\taogangguluwang\test\detection_results_{timestamp}'
os.makedirs(result_folder, exist_ok=True)

# 加载训练好的模型
print("正在加载新训练的模型...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp_new/weights/best.pt')

# 修改类别名显示
model.names[0] = '陶钢过滤网'  # 将索引0的类别名改为"陶钢过滤网"

# 设置置信度阈值
model.conf = 0.3  # 稍微提高置信度阈值
model.iou = 0.45

# 测试多张图片
test_images = ['9.jpg', '14.jpg', '16.jpg', '32.jpg']

for img_name in test_images:
    img_path = f'dataset/images/train/{img_name}'
    
    if os.path.exists(img_path):
        print(f"\n正在测试图片: {img_name}")
        
        # 读取图片
        img = Image.open(img_path)
        
        # 推理
        results = model(img)
        
        # 打印检测结果
        print(f"检测结果:")
        results.print()
        
        # 保存结果
        results.save(save_dir=result_folder)
        print(f"结果已保存到 {result_folder}\\{img_name}")
    else:
        print(f"图片 {img_path} 不存在")

print(f"\n所有测试完成！请查看 {result_folder} 文件夹中的结果图片。")
