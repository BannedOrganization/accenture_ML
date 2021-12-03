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


def f_contour(filename, filename2):

    if filename is not None:
        edged = cv2.Canny(filename, 10, 250)
        #cv2.imwrite("out4/edged4.jpg", edged)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        #cv2.imwrite("out4/closed.jpg", closed)
        cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        total = 0

        for c in cnts:
            if c is None:
                #print("Поиск контуров. Контуры не найдены")
                result5 = None
            else:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.03 * peri, True)
                areapl = cv2.contourArea(approx)
                total += 1
                #print(1010)

                if areapl > 1200 and len(approx) > 3:
                    cr_rect = cv2.boundingRect(c)
                    cv2.drawContours(filename, [c], -1, (255, 0, 0), 1)
                    cv2.imwrite('out4/roi.png', filename[cr_rect[1]:cr_rect[1]+cr_rect[3], cr_rect[0]:cr_rect[0]+cr_rect[2]])
                    crop_image = filename[cr_rect[1]:cr_rect[1]+cr_rect[3], cr_rect[0]:cr_rect[0]+cr_rect[2]]
                    crop_image2 = filename2[cr_rect[1]:cr_rect[1] + cr_rect[3], cr_rect[0]:cr_rect[0] + cr_rect[2]]
                    #result50 = cv2.resize(crop_image, (180, 50))
                    result5 = crop_image2

                else:
                    #print("Поиск контуров. Не найдено похожей на номер области")
                    result5 = None

    else:
        #print("Поиск контуров. Дальнейшие вычисления невозможны")
        result5 = None
        #crop_image = None

    return result5