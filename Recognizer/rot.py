import numpy as np
import scipy.special
import winsound
import math
import matplotlib.pyplot as plt
import sklearn
import time
import scipy.misc
from PIL import Image
import matplotlib.image as mpimg
import os
import json
import cv2


def rot(filename3, filename31, filename312):
    edged = cv2.Canny(filename3, 10, 250)
    linesp = cv2.HoughLinesP(edged, 1, np.pi / 180, 50, None, 50, 10)

    if linesp is not None:
        #for i in range(0, len(linesp)):
            #cv2.imwrite('out4/haf.jpg', filename3)
        #print("есть", len(linesp), "линий Хафа")
        pass

    a = []
    line_norm = []

    if linesp is not None:
        for i in range(0, len(linesp)):
            l1 = linesp[i][0]
            vec = np.array([l1[2] - l1[0], l1[3] - l1[1]])
            horizon = np.array([l1[2] - l1[0], l1[1] - l1[1]])
            #print(cv2.norm(horizon, cv2.NORM_L2) * cv2.norm(vec, cv2.NORM_L2), "делим на это")
            if cv2.norm(horizon, cv2.NORM_L2) * cv2.norm(vec, cv2.NORM_L2) != 0:
                angle = (180.0 / math.pi) * (np.arccos((horizon[0] * vec[0] + horizon[1] * vec[1]) / (
                    cv2.norm(horizon, cv2.NORM_L2) * cv2.norm(vec, cv2.NORM_L2))))

            else:
                angle = 0

            if angle < 46:
                a.insert(i, angle)
                line_norm.insert(i, cv2.norm(vec, cv2.NORM_L2))

        max_line_norm = max(line_norm)  # максимальная длинна вектора
        n = line_norm.index(max_line_norm)  # номер максимального вектора в списке
        angle_final = a[n]
        line_max = linesp[n][0]
        vec_max = np.array([line_max[2] - line_max[0], line_max[3] - line_max[1]])

        #print("Преобразование Хаффа. Максимальная длинна вектора", max_line_norm, "  индекс в списке равен", n,
         #     " угол максимального вектора равен", angle_final)
        #print(a)
        #print("Преобразование Хаффа. Угол поворота равен", angle_final)
        if vec_max[1] < 0:
            result4 = -angle_final
        else:
            result4 = angle_final

    else:
        #print("Преобразование Хаффа. Линии Хаффа не обнаружены, слишком плохое качество фото")
        result4 = 1000

    if result4 == 1000:
        #print("Поворот фото. Дальнейшие вычисления невозможны")
        result2 = None
        result21 = filename31
        result22 = filename312
    else:
        image_center = tuple(np.array(filename3.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, result4, 1.0)
        result2 = cv2.warpAffine(filename3, rot_mat, filename3.shape[1::-1], flags=cv2.INTER_LINEAR)
        result21 = cv2.warpAffine(filename31, rot_mat, filename31.shape[1::-1], flags=cv2.INTER_LINEAR)
        result22 = filename312
    return result2, result21, result22
