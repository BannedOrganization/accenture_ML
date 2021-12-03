import numpy as np
#import scipy.special
import winsound
import math
#import matplotlib.pyplot as plt
#import sklearn
import time
#import tensorflow as tf
#import scipy.misc
#from PIL import Image
#import matplotlib.image as mpimg
import os
#import json
import cv2


def compliment(filename):
    fon = np.zeros((filename.shape[0]+4, filename.shape[1]+4), dtype="uint8")+255
    rb, cb = fon.shape[:2]
    rs, cs = filename.shape[:2]
    top = (rb - rs) / 2
    bottom = rb - rs - top
    left = (cb - cs) / 2
    right = cb - cs - left
    res = cv2.copyMakeBorder(filename, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT,value= (255, 255,255))
    #cv2.imwrite('compliment/fon.jpg', fon)
    return res

def count_black(sourse, first1, first2, second1, second2, tr, order, less):
    cb = []
    if less ==1:
        for i in range(first1, first2):
            counter = 0
            for j in range(second1, second2):
                if order == 1:
                    if sourse[i, j] < tr:
                        counter += 1
                else:
                    if sourse[j, i] < tr:
                        counter += 1

            percent = (counter * 100) / second2
            cb.append(percent)
    else:
        for i in range(first1, first2):
            counter = 0
            for j in range(second1, second2):
                if order == 1:
                    if sourse[i, j] > tr:
                        counter += 1
                else:
                    if sourse[j, i] > tr:
                        counter += 1

            percent = (counter * 100) / second2
            cb.append(percent)

    return cb


def grey_bin(img):
    img0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img1 = cv2.threshold(img0, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img0, img1

def segment(filename4, filename5, filename6, x0, y0):

    ht = 32
    wd = 32
    dim = (ht, wd)
    im0 = filename5
    im1, im2 = grey_bin(im0)
    #cv2.imwrite('seg/filename5-07072019.jpg', filename5)
    height00, width00 = im1.shape
    a =np.asarray(count_black(im1, 0, height00, 0, width00, 180, 1, 1))
    y1 = np.argmin(a[0:round(len(a)/2)])
    #print(y1)
    a1 = a[round(len(a)/2): len(a)+1]
    y02 = np.argmin(a1)
    y2 =y02+round(len(a)/2)
    im_c1 = im0[y1-1:y2+2,0:width00]
    #cv2.imwrite('seg/im_c1.jpg', im0)
    #print(im_c1,'!!!!!!!!!!!!!!!!!!!!!!!!')
    if len(im_c1) !=0:

        im_c1g, im_c1gb = grey_bin(im_c1)
        height000, width000 = im_c1g.shape

        b = np.array(count_black(im_c1g, 0, width000, 0, height000, 180, 0, 1))
        x1 = np.argmax(b[0:round(len(b) / 4)])
        # print(pr)
        # print(b[x1])
        b1 = b[3 * round(len(b) / 4): len(b) + 1]
        x02 = np.argmax(b1)
        x2 = x02 + 3 * round(len(b) / 4)
        im_c = im_c1[0:height000, x1 + 3:x2]
        im_cg, im_b = grey_bin(im_c)
        height0000, width0000 = im_cg.shape
        # cv2.imwrite('seg/crop0707-2019.jpg', im_b)
        pr2 = count_black(im_b, 0, width0000, 0, height0000, 250, 0, 1)
        c = np.array(pr2)
        s_z_c = np.sum(c) / len(c)
        # print(s_z_c,"среднее значение цвета")
        minimum = []
        minimum2 = []
        minimum3 = []
        minimum4 = []
        for i in range(1, width0000 - 1):

            if pr2[i - 1] >= pr2[i] and pr2[i] <= pr2[i + 1] and pr2[i] < s_z_c / 2.3:
                minimum.append(i)
        k = 0
        border = 0

        #if width0000 < 150:
        #    sh = 3
        #else:
        #    sh = 5
        sh = width0000/38

        # print(len(minimum))
        for i in range(1, len(minimum)):
            # im_s1 = cv2.line(im_cg, (minimum[i], 0), (minimum[i], width0000), (0, 255, 0), 1)
            # k += 1
            if minimum[i] - minimum[i - 1] <= sh and i != len(minimum) - 1:
                k += 1
                border += minimum[i]
            else:
                if k != 0:
                    # print(k,"k")
                    border_s = round(border / k)
                    border = 0
                    k = 0
                    minimum2.append(border_s)
                    minimum2.append(minimum[i])
                else:
                    minimum2.append(minimum[i])
        dl = []
        for i in range(1, len(minimum2)):
            dl.append(minimum2[i] - minimum2[i - 1])

        #print(minimum)
        #print(minimum2, "2!!")
        #print(dl, 'Длинны расстояний')
        s_r = np.mean(dl)
        #print(s_r, "среднее расстояние")

        for i in range(0, len(minimum2)):
            if i < round(len(minimum2) / 2):
                delta = 0
                for j in range(3, height0000 - 2):
                    if abs(im_b[j, minimum2[i]] - im_b[j - 1, minimum2[i]]) > 50:
                        delta += 1
                #print(delta, 'delta')
                if abs(minimum2[i] - minimum2[i - 1]) > max(dl) / 2 and delta <= 1:
                    minimum3.append(minimum2[i])
                # im_s2 = cv2.line(im_c, (minimum3[i], 0), (minimum3[i], width0000), (0, 255, 0), 1)
            else:
                delta = 0
                for j in range(3, round(3 * height0000 / 4)):
                    if abs(im_b[j, minimum2[i]] - im_b[j - 1, minimum2[i]]) > 50:
                        delta += 1
                #print(delta, 'delta')
                if abs(minimum2[i] - minimum2[i - 1]) > max(dl) / 2 and delta == 0:
                    minimum3.append(minimum2[i])

        #print(minimum3, "3!!")

        # print(minimum4, "4!!")
        boxes = []
        for i in range(0, len(minimum3) - 1):
            image_crop = im_c[0:0 + height0000, minimum3[i]:minimum3[i + 1] + 1]
            boxes.append(image_crop)

        d = len(boxes)
        #print(d, "количество символов")
        co = 0

        if d > 7 and d < 10:
            a = []
            for i in range(0, len(boxes)):
                segment0 = boxes[i]
                # segment0 = compliment(segment00)
                ht1 = segment0.shape[0]
                wd1 = segment0.shape[1]
                # min_c = np.max(segment0)
                # print(min_c,'mininmum')
                if i > 5:
                    seg_g, seg_bin = grey_bin(segment0)
                    reg = np.asarray(count_black(seg_bin, round(ht1 / 4), round(3 * ht1 / 4), 0, wd1, 200, 1, 0))
                    reg_min = np.argmax(reg)
                    segment0 = segment0[0:round(ht1 / 4) + reg_min, 0:wd1]

                    if i == 6:
                        ht11 = segment0.shape[0]
                        wd11 = segment0.shape[1]
                        seg_g1, seg_bin1 = grey_bin(segment0)
                        reg1 = np.asarray(count_black(seg_bin1, 0, round(wd11 / 2), 0, ht11, 250, 0, 1))
                        reg_max = np.argmax(reg1)
                        segment0 = segment0[0:ht11, reg_max + 2:wd11]
                        # ht1 = segment0.shape[0]
                        # wd1 = segment0.shape[1]
                    # print(ht1, wd1,"новые размеры")
                if ht < ht1 or wd < wd1:
                    segment1 = cv2.resize(segment0, dim, interpolation=cv2.INTER_AREA)
                else:
                    segment1 = cv2.resize(segment0, dim, interpolation=cv2.INTER_LINEAR)
                a.append(segment1)
                #cv2.imwrite('seg/segment1-{0}.jpg'.format(co), segment1)
                co += 1
        else:
            a = []

    else:
        a =[]
        d = 0

    return filename4, d, filename5, filename6, a

