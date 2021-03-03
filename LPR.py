import cv2
import numpy as np
def find_license(rawImg):
    """
    提取车牌
    :param rawImg: 待识别图片
    """
    # 高斯模糊，将图片平滑化，去除干扰的噪声
    img = cv2.GaussianBlur(rawImg, (3, 3), 0)
    cv2.imshow('GaussianBlur', img)
    # 图片灰度化
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imshow('Gray', img)
    # Sobel算子，x方向
    sobelX = cv2.Sobel(img, cv2.CV_16S, 1, 0)

    # sobelY = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    # 转为uint8
    absX = cv2.convertScaleAbs(sobelX)
    cv2.imshow('sobel', absX)
    # absY = cv2.convertScaleAbs(sobelY)
    # dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    img = absX
    # 进行二值化操作
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    # 闭操作，将目标区域连成一个整体，便于后续轮廓的提取
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernelX)
    cv2.imshow('close', img)
    # 膨胀腐蚀 ——> 形态学处理  通过膨胀连接相近的图像区域，通过腐蚀去除孤立细小的色块
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 1))
    img = cv2.dilate(img, kernelX)
    img = cv2.erode(img, kernelX)
    img = cv2.erode(img, kernelY)
    img = cv2.dilate(img, kernelY)
    cv2.imshow('dilate erode', img)
    # 进行平滑处理，中值滤波
    img = cv2.medianBlur(img, 15)
    cv2.imshow('medianBlur', img)
    # 轮廓查找
    contours, w1 = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 对车牌区域进行筛选，小于默认数值的直接排除
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 2000]
    for item in contours:
        rect = cv2.boundingRect(item)
        x = rect[0]
        y = rect[1]
        width = rect[2]
        height = rect[3]
        wh_ratio = width / height
        print('宽高比：', wh_ratio)
        # 要求矩形区域长宽比在2到5.5之间，2到5.5是车牌的长宽比，其余的矩形排除
        if 2 < wh_ratio < 5.5:
            # 裁剪图片
            plate = rawImg[y:y + height, x:x + width]
            cv2.imshow('Plate Region' + str(x), plate)
    # car_contours = []
    # for cnt in contours:
    #     # 框选 生成最小外接矩形 返回值（中心(x,y), (宽,高), 旋转角度）
    #     rect = cv2.minAreaRect(cnt)
    #     area_width, area_height = rect[1]
    #     # 选择宽大于高的区域
    #     if area_width < area_height:
    #         area_width, area_height = area_height, area_width
    #     wh_ratio = area_width / area_height
    #     # 要求矩形区域长宽比在2到5.5之间，2到5.5是车牌的长宽比，其余的矩形排除
    #     if 2 < wh_ratio < 5.5:
    #         car_contours.append(rect)

    # 绘制轮廓
    img = cv2.drawContours(rawImg, contours, -1, (0, 0, 255), 2)
    cv2.imshow("Img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image = cv2.imread("./Test/0.jpg")
    find_license(image)
