import os
from PIL import Image

def convert_to_jpg(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 检查文件是否为图片
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif')):
            # 打开图片
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)

            # 保存为JPG格式
            jpg_filename = os.path.splitext(filename)[0] + '.jpg'
            jpg_path = os.path.join(output_folder, jpg_filename)
            img.save(jpg_path, 'JPEG')

            print(f"Converted {filename} to {jpg_filename}")

# 示例用法
input_folder = 'E:/programming/code/resnet/Brain Tumor Data Set/Brain Tumor Data Set/Healthy'
output_folder = 'E:/programming/code/resnet/Brain Tumor Data Set/Brain Tumor preprocessed/Healthy'
convert_to_jpg(input_folder, output_folder)