import re
import os
import cv2
import shutil
import numpy as np


# 检测输入图像是否需要
def check_img(img_path):
    '''
    从文件大小、分辨率、梯度三个方面进行检测
    :param img_path:
    :return:
    '''
    img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
    # file info：文件大小、分辨率、长宽比
    file_size = os.path.getsize(img_path)  # 获取文件大小
    img_height, img_width = img.shape[:2]  # 获取图片分辨率
    if file_size < 10 * 1024 or img_width < 256 or img_height < 256:
        return False

    # image basic feature：图像颜色和纹理信息：纹理图像梯度
    img_dy = img[:img_height-1] - img[1:]
    img_dx = img[:, :img_width-1] - img[:, 1:]
    img_gradient = np.mean(np.abs(img_dx)) + np.mean(np.abs(img_dy))
    print(img_path, "img_gradient =", img_gradient)
    if img_gradient < 50:
        return False
    return True

def bright_check(img_path):
    '''
    从亮度进行检测
    :param img_path:图片路径
    :return:bool值
    '''
    # 把图片转换为单通道的灰度图
    img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 获取形状以及长宽
    img_shape = gray_img.shape
    height, width = img_shape[0], img_shape[1]
    size = gray_img.size
    # 灰度图的直方图
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    # 计算灰度图像素点偏离均值(128)程序
    a = 0
    ma = 0
    reduce_matrix = np.full((height, width), 128)
    shift_value = gray_img - reduce_matrix
    shift_sum = sum(map(sum, shift_value))
    da = shift_sum / size
    # 计算偏离128的平均偏差
    for i in range(256):
        ma += (abs(i - 128 - da) * hist[i])
    m = abs(ma / size)
    # 亮度系数
    k = abs(da) / m
    if k[0] > 1:
        # 过亮
        # if da > 0:
        #     print("过亮")
        # else:
        #     print("过暗")
        return False
    else:
        # print("亮度正常")
        return True

def check_contrast(img_path):
    '''
    从对比度方面进行检测
    :param img_path:图片路径
    :return:是否对比度低的bool值
    '''
    from skimage import data, exposure, io
    img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
    result = exposure.is_low_contrast(img)
    return result


if __name__ == '__main__':
    root_dir = "../Image-Downloader-master/download_images/car"
    file_suffix = "jpeg|jpg|png|bmp"
    remove_dir = root_dir + "/remove"
    if not os.path.exists(remove_dir):
        os.makedirs(remove_dir)
    for img_name in os.listdir(root_dir):
        # 对处理文件的类型进行过滤
        if re.search(file_suffix, img_name) is None:
            continue
        img_path = root_dir + "/" + img_name
        if not check_img(img_path) or not bright_check(img_path) or check_contrast(img_path):
            output_path = remove_dir + "/" + img_name
            shutil.move(img_path, output_path)

