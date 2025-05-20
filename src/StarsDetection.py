# File      :StarsDetection.py
# Author    :Wu Ji
# Time      :2025/02/07
# Version   :1.0
# Function  :

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from icecream import ic
def detect_stars(image_path, visualize=True):
    """
    检测图片中的星星位置，并可选择显示处理过程。

    Args:
        image_path (str): 图片文件路径，支持常见格式（如jpg、png）。
        visualize (bool): 是否显示中间处理结果，默认为True。

    Returns:
        star_positions (list of tuples): 检测到的星星坐标列表，每个元素为(y, x)。
        processed_image (numpy.ndarray): 处理后的图像，标记了检测到的星星。
    """
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("无法读取图片文件，请检查路径是否正确。")

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 增强对比度和亮度（可调整alpha和beta）
    alpha = 1.5  # 对比度增强系数，大于1时增加对比度
    beta = 30  # 亮度增强量，根据需要调整
    enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # 应用高斯模糊去噪（可调整核大小）
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # 使用自适应阈值分割提取星星区域
    blockSize = 11  # 块大小，必须是奇数且大于1
    C = 5  # 减去的常数值，调整检测灵敏度
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,  # 使用均值方法
        thresholdType=cv2.THRESH_BINARY_INV,  # 反二值化（将星星设为白色）
        blockSize=blockSize,
        C=C
    )

    # 形态学开运算去噪并优化形状
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # 检测星星位置（白色区域的坐标）
    rows, cols = np.where(opening == 255)
    star_positions = list(zip(rows, cols))

    # 可视化结果
    if visualize:
        plt.figure(figsize=(14, 8))

        plt.subplot(231), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('原始图像')
        plt.xticks([]), plt.yticks([])

        plt.subplot(232), plt.imshow(gray, cmap='gray'), plt.title('灰度图')
        plt.xticks([]), plt.yticks([])

        plt.subplot(233), plt.imshow(enhanced, cmap='gray'), plt.title('对比度增强')
        plt.xticks([]), plt.yticks([])

        plt.subplot(234), plt.imshow(blurred, cmap='gray'), plt.title('高斯模糊去噪')
        plt.xticks([]), plt.yticks([])

        plt.subplot(235), plt.imshow(adaptive_thresh, cmap='gray'), plt.title('自适应二值化')
        plt.xticks([]), plt.yticks([])

        plt.subplot(236), plt.imshow(opening, cmap='gray'), plt.title('形态学开运算优化')
        plt.xticks([]), plt.yticks([])

        plt.tight_layout()
        plt.show()

    # 绘制检测到的星星
    processed_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    for y, x in star_positions:
        cv2.circle(processed_image, (x, y), 3, (0, 255, 0), -1)  # 绿色圆圈标记

    return star_positions, processed_image


# 示例用法
if __name__ == "__main__":


    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 检测图片中的星星
    image_path = os.path.join(current_dir, r"..\data\smalltestdata.png")
    star_positions, processed_image = detect_stars(image_path, visualize=True)

    print(f"检测到 {len(star_positions)} 颗星星。")
    plt.figure(figsize=(10, 6))
    plt.imshow(processed_image), plt.title('最终结果')
    plt.xticks([]), plt.yticks([])
    plt.show()

    ic(star_positions)
