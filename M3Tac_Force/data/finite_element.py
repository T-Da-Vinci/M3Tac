import math
import pandas
import matplotlib.pyplot as plt
import numpy as np
import cv2
from glob import glob


def read_table(filename):
    data = pandas.read_table(filename, header=None, delim_whitespace=True)
    return data


def finite_element(type=None):
    force_xyz = "z"
    data = read_table("force_map//0.5.txt")
    if force_xyz == "x" or force_xyz == "y":
        idx = 2
    else:
        idx = 3
    res = []
    k = len(data[idx].values)
    for i in range(0, k):
        if float(data[idx].values[i]) != 0:
            num = round(data[idx].values[i] / 1000, 4)
            res.append(num)
    m = len(res)
    id = np.arange(m)
    fit_id = np.arange(-(m - 1) / 2, (m + 1) / 2)
    ####################拟合方程##############
    z1 = np.polyfit(fit_id, res, 10)
    p1 = np.poly1d(z1)
    model_img = np.zeros((m, m))
    ####################利用距离中心点的距离来求和##############
    for i in range(0, m):
        for j in range(0, m):
            dis = math.sqrt(abs((i - m / 2) * (i - m / 2)) + abs((j - m / 2) * (j - m / 2)))
            if dis > (m - 1) / 2:
                model_img[i][j] = 0
            else:
                model_img[i][j] = abs(p1(dis))
    ##########测试拟合效果#############
    XX = np.arange(-(m) / 2, (m) / 2, 1)
    YY = np.arange(-(m) / 2, (m) / 2, 1)
    X, Y = np.meshgrid(XX, YY)
    ax3 = plt.axes(projection='3d')
    ax3.plot_surface(X, Y, model_img, cmap='rainbow')
    plt.show()

    #########################读取图像数据和力数据########
    image_list = sorted(glob("force_map_finite_element\\test0.5\\rgb_mask\\*.jpg"))
    f = open("force_map_finite_element\\test0.5\\data_force.txt")

    for i in range(0, len(image_list)):
        line = f.readline()
        line = line[1:len(line) - 2]
        array = line.split(',')
        img = cv2.imread(image_list[i])
        #########################从测量的力提取对应方向的力F###############
        if force_xyz == "x":
            f_id = 0  # 0表示x
        elif force_xyz == "y":
            f_id = 1  # 1表示y
        else:
            f_id = 2  # 2表示z
        force = float(array[f_id])
        #########################圆形检测###########
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
        mask = thresh.copy()
        contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < 200:
                continue
            else:

                (x, y), radius = cv2.minEnclosingCircle(c)  # 找到最小圆，并返回圆心坐标和半径
                center = (int(x), int(y))
                radius = int(radius)
                img = cv2.circle(img, center, radius, (255, 255, 255), -1)
                cv2.fillPoly(mask, pts=[c], color=255)
        ######################获取圆的最小外接矩形####
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
        contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv2.contourArea(c) < 200:
                continue
            x, y, w, h = cv2.boundingRect(c)
            ####################将矩形的最长边作为裁剪填充区域的边界######
            if w > h:
                h = w
            else:
                w = h
            ##################将最开始生成的map修改尺寸为填充区域的尺寸
            resized = cv2.resize(model_img, (w, w), interpolation=cv2.INTER_AREA)
            # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)#确定对角线然后画出矩阵
            ###########取最大值########
            npmax = np.max(model_img)
            ###########将圆形区域用map填充
            img1[y:y + w, x:x + w] = resized
            ##########归一化########
            img1 = img1 / npmax * force
            #########plot绘制结果##########
            XX = np.arange(0, 300, 1)
            YY = np.arange(0, 300, 1)
            X, Y = np.meshgrid(XX, YY)
            ax3 = plt.axes(projection='3d')
            ax3.plot_surface(X, Y, img1, cmap='rainbow')
            plt.show()


if __name__ == '__main__':
    data_path = "/home/shoujie/Program/force_map_new/dataset/force_map_finite_element"
