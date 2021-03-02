import cv2
import numpy as np
import os
import time
from SVM_Train import SVM
import SVM_Train
from args import args


def imread(filename):
    """
    图片读取
    :param filename: 文件路径
    :return:
    """
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)

def point_limit(point):
    """
    缩小边界
    :param point:
    """
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0

class PlateRecognition:
    """
    车牌识别类
    """
    # 初始化相关属性
    def __init__(self):
        self.SZ = args.Size  # 训练图片长宽
        self.MAX_WIDTH = args.MAX_WIDTH  # 原始图片最大宽度
        self.Min_Area = args.Min_Area  # 车牌区域允许最大面积
        self.PROVINCE_START = args.PROVINCE_START
        self.provinces = args.provinces
        self.cardType = args.cardType
        self.Prefecture = args.Prefecture
        self.config = args.Pic_config
        # 训练模型  默认参数C=1，gamma=0.5
        self.model = SVM()
        self.cn_model = SVM()

    def __accurate_place(self, card_img_hsv, limit1, limit2, color):
        row_num, col_num = card_img_hsv.shape[:2]
        xl = col_num
        xr = 0
        yh = 0
        yl = row_num
        # col_num_limit = self.cfg["col_num_limit"]
        row_num_limit = self.config["row_num_limit"]
        col_num_limit = col_num * 0.8 if color != "green" else col_num * 0.5  # 绿色有渐变
        for i in range(row_num):
            count = 0
            for j in range(col_num):
                H = card_img_hsv.item(i, j, 0)
                S = card_img_hsv.item(i, j, 1)
                V = card_img_hsv.item(i, j, 2)
                if limit1 < H <= limit2 and 34 < S and 46 < V:
                    count += 1
            if count > col_num_limit:
                if yl > i:
                    yl = i
                if yh < i:
                    yh = i
        for j in range(col_num):
            count = 0
            for i in range(row_num):
                H = card_img_hsv.item(i, j, 0)
                S = card_img_hsv.item(i, j, 1)
                V = card_img_hsv.item(i, j, 2)
                if limit1 < H <= limit2 and 34 < S and 46 < V:
                    count += 1
            if count > row_num - row_num_limit:
                if xl > j:
                    xl = j
                if xr < j:
                    xr = j
        return xl, xr, yh, yl

    # 确定车牌区域
    def __find_location(self, src):
        if type(src) == type(""):
            img = imread(src)
        else:
            img = src
        pic_height, pic_width = img.shape[:2]
        if pic_width > self.MAX_WIDTH:
            resize_rate = self.MAX_WIDTH / pic_width
            img = cv2.resize(img, (self.MAX_WIDTH, int(pic_height * resize_rate)),
                             interpolation=cv2.INTER_AREA)  # 图片分辨率调整
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 定义一个核
        img = cv2.filter2D(img, -1, kernel=kernel)  # 锐化
        blur = self.config["blur"]
        # 高斯去噪
        if blur > 0:
            img = cv2.GaussianBlur(img, (blur, blur), 0)
        oldimg = img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('GaussianBlur', img)
        kernel = np.ones((20, 20), np.uint8)
        img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # 开运算
        img_opening = cv2.addWeighted(img, 1, img_opening, -1, 0)  # 与上一次开运算结果融合
        cv2.imshow('img_opening', img_opening)

        # 找到图像边缘
        ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 二值化
        img_edge = cv2.Canny(img_thresh, 100, 200)
        cv2.imshow('img_edge', img_edge)
        # 使用开运算和闭运算让图像边缘成为一个整体
        kernel = np.ones((self.config["morphologyr"], self.config["morphologyc"]), np.uint8)
        img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)  # 闭运算
        img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)  # 开运算
        cv2.imshow('img_edge2', img_edge2)
        # 查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
        try:
            # 由于不同版本的opencv返回值数量不同，因此用try catch进行处理
            image, contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            # ValueError: not enough values to unpack (expected 3, got 2)
            # cv2.findContours方法在高版本OpenCV中只返回两个参数
            contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.Min_Area]
        # 排除不是车牌的矩形区域
        car_contours = []
        for cnt in contours:
            # 框选 生成最小外接矩形 返回值（中心(x,y), (宽,高), 旋转角度）
            rect = cv2.minAreaRect(cnt)
            # print('宽高:',rect[1])
            area_width, area_height = rect[1]
            # 选择宽大于高的区域
            if area_width < area_height:
                area_width, area_height = area_height, area_width
            wh_ratio = area_width / area_height
            # print('宽高比：',wh_ratio)
            # 要求矩形区域长宽比在2到5.5之间，2到5.5是车牌的长宽比，其余的矩形排除
            if 2 < wh_ratio < 5.5:
                car_contours.append(rect)
            # 框出所有可能的矩形
            # oldimg = cv2.drawContours(oldimg, contours, -1, (0, 0, 255), 2)

        # 矩形区域可能是倾斜的矩形，需要矫正，以便使用颜色定位
        card_imgs = []
        for rect in car_contours:
            # 创造角度，使得左、高、右、低拿到正确的值
            if -1 < rect[2] < 1:
                angle = 1
            else:
                angle = rect[2]
            # 扩大范围，避免车牌边缘被排除
            rect = (rect[0], (rect[1][0] + 5, rect[1][1] + 5), angle)
            box = cv2.boxPoints(rect)
            height_point = right_point = [0, 0]
            left_point = low_point = [pic_width, pic_height]
            for point in box:
                if left_point[0] > point[0]:
                    left_point = point
                if low_point[1] > point[1]:
                    low_point = point
                if height_point[1] < point[1]:
                    height_point = point
                if right_point[0] < point[0]:
                    right_point = point
            # 正角度
            if left_point[1] <= right_point[1]:
                new_right_point = [right_point[0], height_point[1]]
                # 字符只是高度需要改变
                pts2 = np.float32([left_point, height_point, new_right_point])
                pts1 = np.float32([left_point, height_point, right_point])
                M = cv2.getAffineTransform(pts1, pts2)
                dst = cv2.warpAffine(oldimg, M, (pic_width, pic_height))
                point_limit(new_right_point)
                point_limit(height_point)
                point_limit(left_point)
                card_img = dst[int(left_point[1]):int(height_point[1]), int(left_point[0]):int(new_right_point[0])]
                card_imgs.append(card_img)
            # 负角度
            elif left_point[1] > right_point[1]:
                new_left_point = [left_point[0], height_point[1]]
                # 字符只是高度需要改变
                pts2 = np.float32([new_left_point, height_point, right_point])
                pts1 = np.float32([left_point, height_point, right_point])
                M = cv2.getAffineTransform(pts1, pts2)
                dst = cv2.warpAffine(oldimg, M, (pic_width, pic_height))
                point_limit(right_point)
                point_limit(height_point)
                point_limit(new_left_point)
                card_img = dst[int(right_point[1]):int(height_point[1]), int(new_left_point[0]):int(right_point[0])]
                card_imgs.append(card_img)
        cv2.imshow('angel card_img', card_imgs[0])
        #____开始使用颜色定位，排除不是车牌的矩形，目前只识别蓝、绿、黄车牌
        colors = []
        for card_index, card_img in enumerate(card_imgs):
            green = yellow = blue = black = white = 0
            try:
                # 有转换失败的可能，原因来自于上面矫正矩形出错
                card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
            except:
                card_img_hsv = None
            if card_img_hsv is None:
                continue
            row_num, col_num = card_img_hsv.shape[:2]
            card_img_count = row_num * col_num
            # 确定车牌颜色
            for i in range(row_num):
                for j in range(col_num):
                    H = card_img_hsv.item(i, j, 0)
                    S = card_img_hsv.item(i, j, 1)
                    V = card_img_hsv.item(i, j, 2)
                    if 11 < H <= 34 and S > 34:  # 图片分辨率调整
                        yellow += 1
                    elif 35 < H <= 99 and S > 34:  # 图片分辨率调整
                        green += 1
                    elif 99 < H <= 124 and S > 34:  # 图片分辨率调整
                        blue += 1

                    if 0 < H < 180 and 0 < S < 255 and 0 < V < 46:
                        black += 1
                    elif 0 < H < 180 and 0 < S < 43 and 221 < V < 225:
                        white += 1
            color = "no"
            # print('黄：{:<6}绿：{:<6}蓝：{:<6}'.format(yellow,green,blue))
            limit1 = limit2 = 0
            if yellow * 2 >= card_img_count:
                color = "yellow"
                limit1 = 11
                limit2 = 34  # 有的图片有色偏偏绿
            elif green * 2 >= card_img_count:
                color = "green"
                limit1 = 35
                limit2 = 99
            elif blue * 2 >= card_img_count:
                color = "blue"
                limit1 = 100
                limit2 = 124  # 有的图片有色偏偏紫
            elif black + white >= card_img_count * 0.7:
                color = "bw"
            # print(color)
            colors.append(color)
            # print(blue, green, yellow, black, white, card_img_count)
            if limit1 == 0:
                continue
            # 根据车牌颜色再定位，缩小边缘非车牌边界
            xl, xr, yh, yl = self.__accurate_place(card_img_hsv, limit1, limit2, color)
            if yl == yh and xl == xr:
                continue
            need_accurate = False
            if yl >= yh:
                yl = 0
                yh = row_num
                need_accurate = True
            if xl >= xr:
                xl = 0
                xr = col_num
                need_accurate = True
            card_imgs[card_index] = card_img[yl:yh, xl:xr] \
                if color != "green" or yl < (yh - yl) // 4 \
                else card_img[yl - (yh - yl) // 4:yh, xl:xr]
            if need_accurate:  # 可能x或y方向未缩小，需要再试一次
                card_img = card_imgs[card_index]
                card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
                xl, xr, yh, yl = self.__accurate_place(card_img_hsv, limit1, limit2, color)
                if yl == yh and xl == xr:
                    continue
                if yl >= yh:
                    yl = 0
                    yh = row_num
                if xl >= xr:
                    xl = 0
                    xr = col_num
            card_imgs[card_index] = card_img[yl:yh, xl:xr] \
                if color != "green" or yl < (yh - yl) // 4 else card_img[yl - (yh - yl) // 4:yh, xl:xr]
        cv2.imshow("result", card_imgs[0])
        cv2.waitKey(0)
        # cv2.imwrite('1.jpg', card_imgs[0])
        print('颜色识别结果：' + colors[0])
        return card_imgs, colors

    # 利用投影法，根据设定的阈值和图片直方图，找出波峰，用于分隔字符
    def __find_waves(self, threshold, histogram):
        up_point = -1  # 上升点
        is_peak = False
        if histogram[0] > threshold:
            up_point = 0
            is_peak = True
        wave_peaks = []
        for i, x in enumerate(histogram):
            if is_peak and x < threshold:
                if i - up_point > 2:
                    is_peak = False
                    wave_peaks.append((up_point, i))
            elif not is_peak and x >= threshold:
                is_peak = True
                up_point = i
        if is_peak and up_point != -1 and i - up_point > 4:
            wave_peaks.append((up_point, i))
        return wave_peaks

    # 根据找出的波峰，分隔图片，从而得到逐个字符图片
    def __seperate_card(self, img, waves):
        part_cards = []
        for wave in waves:
            part_cards.append(img[:, wave[0]:wave[1]])
        return part_cards

    def __identify_plate(self, card_imgs, colors, model, cn_model):
        # 识别车牌中的字符
        result = {}
        predict_result = []
        roi = None
        card_color = None
        for i, color in enumerate(colors):
            if color in ("blue", "yellow", "green"):
                card_img = card_imgs[i]
                # old_img = card_img
                # 做一次锐化处理
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
                card_img = cv2.filter2D(card_img, -1, kernel=kernel)
                cv2.imshow("custom_blur", card_img)
                cv2.waitKey(0)
                # RGB转GARY
                gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
                cv2.imshow('gray_img', gray_img)
                cv2.waitKey(0)
                # 黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反向
                if color == "green" or color == "yellow":
                    gray_img = cv2.bitwise_not(gray_img)
                # 二值化
                ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                cv2.imshow('gray_img', gray_img)
                cv2.waitKey(0)
                # 查找水平直方图波峰
                x_histogram = np.sum(gray_img, axis=1)
                # 最小值
                x_min = np.min(x_histogram)
                # 均值
                x_average = np.sum(x_histogram) / x_histogram.shape[0]
                x_threshold = (x_min + x_average) / 2
                wave_peaks = self.__find_waves(x_threshold, x_histogram)
                if len(wave_peaks) == 0:
                    continue

                # 认为水平方向，最大的波峰为车牌区域
                wave = max(wave_peaks, key=lambda x: x[1] - x[0])
                gray_img = gray_img[wave[0]:wave[1]]
                cv2.imshow('gray_img', gray_img)
                cv2.waitKey(0)

                # 查找垂直直方图波峰
                row_num, col_num = gray_img.shape[:2]
                # 去掉车牌上下边缘1个像素，避免白边影响阈值判断
                gray_img = gray_img[1:row_num - 1]
                cv2.imshow('gray_img', gray_img)
                cv2.waitKey(0)
                y_histogram = np.sum(gray_img, axis=0)
                y_min = np.min(y_histogram)
                y_average = np.sum(y_histogram) / y_histogram.shape[0]
                y_threshold = (y_min + y_average) / 5  # U和0要求阈值偏小，否则U和0会被分成两半

                wave_peaks = self.__find_waves(y_threshold, y_histogram)
                # print(wave_peaks)

                # for wave in wave_peaks:
                #	cv2.line(card_img, pt1=(wave[0], 5), pt2=(wave[1], 5), color=(0, 0, 255), thickness=2)
                # 车牌字符数应大于6
                if len(wave_peaks) <= 6:
                    #   print(wave_peaks)
                    continue

                wave = max(wave_peaks, key=lambda x: x[1] - x[0])
                max_wave_dis = wave[1] - wave[0]
                # 判断是否是左侧车牌边缘
                if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 and wave_peaks[0][0] == 0:
                    wave_peaks.pop(0)

                # 组合分离汉字
                cur_dis = 0
                for i, wave in enumerate(wave_peaks):
                    if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
                        break
                    else:
                        cur_dis += wave[1] - wave[0]
                if i > 0:
                    wave = (wave_peaks[0][0], wave_peaks[i][1])
                    wave_peaks = wave_peaks[i + 1:]
                    wave_peaks.insert(0, wave)

                # 去除车牌上的分隔点
                point = wave_peaks[2]
                if point[1] - point[0] < max_wave_dis / 3:
                    point_img = gray_img[:, point[0]:point[1]]
                    if np.mean(point_img) < 255 / 5:
                        wave_peaks.pop(2)

                if len(wave_peaks) <= 6:
                    # print("peak less 2:", wave_peaks)
                    continue
                # print(wave_peaks)
                # 分割牌照字符
                part_cards = self.__seperate_card(gray_img, wave_peaks)

                # 分割输出
                #for i, part_card in enumerate(part_cards):
                #    cv2.imshow(str(i), part_card)

                # 识别
                for i, part_card in enumerate(part_cards):
                    # 可能是固定车牌的铆钉
                    if np.mean(part_card) < 255 / 5:
                        continue
                    part_card_old = part_card
                    w = abs(part_card.shape[1] - self.SZ) // 2

                    # 边缘填充
                    part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    cv2.imshow('part_card', part_card)
                    cv2.waitKey(0)
                    # 图片缩放（缩小）
                    part_card = cv2.resize(part_card, (self.SZ, self.SZ), interpolation=cv2.INTER_AREA)
                    cv2.imshow('part_card', part_card)
                    cv2.waitKey(0)
                    part_card = SVM_Train.preprocess_hog([part_card])

                    if i == 0:  # 识别汉字
                        resp = self.cn_model.predict(part_card)  # 匹配样本
                        character = self.provinces[int(resp[0]) - self.PROVINCE_START]
                        print(character)
                    else:  # 识别字母
                        resp = self.model.predict(part_card)  # 匹配样本
                        character = chr(resp[0])
                        print(character)
                    # 判断最后一个数是否是车牌边缘，假设车牌边缘被认为是1
                    if character == "1" and i == len(part_cards) - 1:
                        if color == 'blue' and len(part_cards) > 7:
                            if part_card_old.shape[0] / part_card_old.shape[1] >= 7:  # 1太细，认为是边缘
                                continue
                        elif color == 'blue' and len(part_cards) > 7:
                            if part_card_old.shape[0] / part_card_old.shape[1] >= 7:  # 1太细，认为是边缘
                                continue
                        elif color == 'green' and len(part_cards) > 8:
                            if part_card_old.shape[0] / part_card_old.shape[1] >= 7:  # 1太细，认为是边缘
                                continue
                    predict_result.append(character)
                roi = card_img  # old_img
                card_color = color
                break
        return predict_result, roi, card_color  # 识别到的字符、定位的车牌图像、车牌颜色

    def __start_recognize__(self, src):
        result = {}
        start = time.time()
        if os.path.exists("svm.dat"):
            self.model.load("svm.dat")
        else:
            raise FileNotFoundError("svm.dat")
        if os.path.exists("svmchinese.dat"):
            self.cn_model.load("svmchinese.dat")
        else:
            raise FileNotFoundError("svmchinese.dat")

        card_imgs, colors = self.__find_location(src)
        if card_imgs is []:
            print('车牌区域提取为空（card_imgs）')
            return
        else:
            predict_result, roi, card_color = self.__identify_plate(card_imgs, colors, self.model, self.cn_model)
            if predict_result:
                result['UseTime'] = round((time.time() - start), 2)
                result['InputTime'] = time.strftime("%Y-%m-%d %H:%M:%S")
                result['Type'] = self.cardType[card_color]
                result['List'] = predict_result
                result['Number'] = ''.join(predict_result[:2]) + '·' + ''.join(predict_result[2:])
                result['Picture'] = roi
                try:
                    result['From'] = ''.join(self.Prefecture[result['List'][0]][result['List'][1]])
                except:
                    result['From'] = '未知'
                return result
            else:
                return None

if __name__ == '__main__':
    c = PlateRecognition()
    result = c.__start_recognize__("./Test/5.jpg")
    print(result)