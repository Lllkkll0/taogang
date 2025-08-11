import os
import xml.etree.ElementTree as ET

def convert_xml_to_yolo(xml_file, classes):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # 获取图片尺寸
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)
    
    yolo_annotations = []
    
    # 遍历所有object
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in classes:
            continue
            
        class_id = classes.index(class_name)
        
        # 获取边界框坐标
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        # 转换为YOLO格式 (归一化的中心点坐标和宽高)
        x_center = (xmin + xmax) / 2.0 / width
        y_center = (ymin + ymax) / 2.0 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        
        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    
    return yolo_annotations

def main():
    # 定义类别（根据你的实际情况修改）
    classes = ['陶钢过滤网']  # 根据XML中的实际类别名修改
    
    # 输入和输出目录
    xml_dir = 'dataset/labels/train'
    output_dir = 'dataset/labels/train'
    
    # 遍历所有XML文件
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(xml_dir, xml_file)
            
            # 转换
            yolo_annotations = convert_xml_to_yolo(xml_path, classes)
            
            # 保存为txt文件
            txt_file = xml_file.replace('.xml', '.txt')
            txt_path = os.path.join(output_dir, txt_file)
            
            with open(txt_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            
            print(f"转换完成: {xml_file} -> {txt_file}")

if __name__ == '__main__':
    main()
