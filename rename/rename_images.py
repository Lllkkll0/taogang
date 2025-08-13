import os
import sys
import argparse

def rename_images(folder_path, start_num, end_num):
    # 支持的图片扩展名
    image_exts = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    
    # 验证文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在")
        return False

    # 获取所有图片文件
    image_files = []
    for f in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, f)):
            ext = os.path.splitext(f)[1].lower()
            if ext in image_exts:
                image_files.append(f)
    
    # 按文件名排序
    image_files.sort()
    total_files = len(image_files)
    
    # 检查文件数量是否匹配
    expected_count = end_num - start_num + 1
    if total_files != expected_count:
        print(f"警告: 文件夹内有 {total_files} 张图片，但指定的数字范围({start_num}-{end_num})需要 {expected_count} 个文件")
        if input("是否继续? (y/n): ").lower() != 'y':
            return False
    
    # 临时重命名阶段 (防止覆盖)
    temp_files = []
    for idx, filename in enumerate(image_files):
        file_ext = os.path.splitext(filename)[1]
        temp_name = f"temp_{idx}{file_ext}"
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, temp_name)
        os.rename(src, dst)
        temp_files.append((temp_name, file_ext))
    
    # 最终重命名阶段
    for idx, (temp_name, file_ext) in enumerate(temp_files):
        new_name = f"{start_num + idx}{file_ext}"
        src = os.path.join(folder_path, temp_name)
        dst = os.path.join(folder_path, new_name)
        os.rename(src, dst)
        print(f"重命名: {temp_name} -> {new_name}")
    
    print(f"成功重命名 {len(temp_files)} 个文件")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='批量重命名图片文件')
    parser.add_argument('folder', help='图片文件夹路径')
    parser.add_argument('start', type=int, help='起始数字')
    parser.add_argument('end', type=int, help='结束数字')
    
    args = parser.parse_args()
    
    if not rename_images(args.folder, args.start, args.end):
        sys.exit(1)