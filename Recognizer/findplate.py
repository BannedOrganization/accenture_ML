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


def findplate(filename):

    if filename is not None:
        ret, gray0 = cv2.threshold(filename, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        gray = cv2.GaussianBlur(gray0, (3, 3), 0)
        edged = cv2.Canny(gray, 10, 250)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        total = 0
        pl_n = []
        sh = -1
        for c in cnts:
            if c is None:
                pass
                #print("Поиск контура в номере. Контуры в области рассматриваемой как номер не найдены")
                #result60 = 0
            else:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.03 * peri, True)
                #areapl = cv2.contourArea(approx)

                if len(approx) == 4:
                    #result60 = 1
                    total += 1
                    sh += 1
                    pl_n.insert(sh, filename)
                    #cv2.imwrite('out4/plate-c{0}.jpg'.format(total), filename)
                    #print("прямоугольников найдено", len(pl_n))

                else:
                    #print("Поиск контура в номере. Не найдено похожей на номер области")
                    #result60 = 0
                    pass

    else:
        #print("Поиск контура в номере. Дальнейшие вычисления невозможны")
        #result60 = 0
        pl_n = []
    return pl_n