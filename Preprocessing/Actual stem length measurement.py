import cv2
import numpy as np

def measure_stem_length(image_path, scale_factor):
    # 加载图像
    segmented_image = cv2.imread(image_path)

    # 定义茎部的颜色阈值范围（根据分割结果进行调整）
    lower_green = np.array([0, 100, 0], dtype=np.uint8)
    upper_green = np.array([100, 255, 100], dtype=np.uint8)

    # 颜色阈值化
    mask = cv2.inRange(segmented_image, lower_green, upper_green)

    # 进行形态学操作，填充孔洞
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到茎部轮廓
    stem_contour = max(contours, key=cv2.contourArea)

    # 找到茎部轮廓的最上端和最下端点
    top_point = tuple(stem_contour[stem_contour[:, :, 1].argmin()][0])
    bottom_point = tuple(stem_contour[stem_contour[:, :, 1].argmax()][0])

    # 计算茎长的估计值
    stem_length = abs(bottom_point[1] - top_point[1]) * scale_factor

    return stem_length

import os

# 图片文件夹路径
image_folder = r'C:\Users\10691\Desktop\stem\2p'

# 尺度因子
scale_factor = 0.016

# 结果文件路径
output_file = r'C:\Users\10691\Desktop\stem\s-l-s.txt'

# 打开结果文件以写入数据
with open(output_file, 'w') as f:
    # 遍历图片文件夹中的所有图片文件
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # 构建图片文件的完整路径
            image_path = os.path.join(image_folder, filename)

            # 调用函数进行处理
            stem_length = measure_stem_length(image_path, scale_factor)

            # 输出茎长的估计值
            f.write(f'{filename}: {stem_length:.2f}\n')