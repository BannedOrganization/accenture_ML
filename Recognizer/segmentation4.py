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
import imutils
from skimage.transform import resize
from skimage.filters import threshold_local
from skimage import segmentation
from skimage import measure
import treshhold_ots


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





def segmentation_s(filename4,filename5, filename6, x0, y0):
#def f_plt(filename, filename2, filename3):

    if filename4 is not None:
        im0 = cv2.threshold(cv2.cvtColor(filename4, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

        #im0 = filename4
        H = filename4.shape[0]
        W = filename4.shape[1]

        ht = 32
        wd = 32

        dim = (ht, wd)

        #V = cv2.split(cv2.cvtColor(im0, cv2.COLOR_BGR2HSV))[2]
        #T = threshold_local(V, 29, offset=15, method="gaussian")
        #im1 = (V > T).astype("uint8") * 255

        im1 = cv2.bitwise_not(im0)
        #cv2.imwrite("out4/bitwise.jpg", im1)
        labels = measure.label(im1, neighbors=8, background=0)
        charCandidates = np.zeros(im1.shape, dtype="uint8")
        im33 = filename4
        #print(labels.shape)
        boxes = []
        segments = {}
        ind = 0
        cntrs = []
        cntrs.insert(ind, (0, 0, 0, 0))

        for label in np.unique(labels):
            # if this is the background label, ignore it
            if label == 0:
                continue

            # otherwise, construct the label mask to display only connected components for the
            # current label, then find contours in the label mask
            labelMask = np.zeros(im1.shape, dtype="uint8")
            labelMask[labels == label] = 255
            cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #cnts = cnts[0] if imutils.is_cv2() else cnts[1]
            cnts = cnts[0]

            if len(cnts) > 0:
                # grab the largest contour which corresponds to the component in the mask, then
                # grab the bounding box for the contour

                c = max(cnts, key=cv2.contourArea)
                ind += 1
                #print(len(c),"количество контуров")
                (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
                cntrs.insert(ind, (boxX, boxY, boxW, boxH))
                #print(len(cntrs),"8888")
                #print(ind, (cntrs[ind])[0])

                if boxW*boxH > 100 and boxW < W/5 and boxW > 5 and boxY+boxH > round(H/2) and boxY<round(H/2) :
                    #cv2.rectangle(im33, (boxX, boxY), (boxX + boxW, boxY + boxH), (0, 0, 255), 2)
                    # compute the aspect ratio, solidity, and height ratio for the component
                    if x0 is not None and y0 is not None:
                        cv2.rectangle(filename6, (boxX + x0-1 , boxY + y0-2), (boxX + x0 + boxW+2, boxY + y0 + boxH+4 ), (0, 255, 0), 2)

                    boxes.append((boxX, boxY, boxW, boxH))

                    aspectRatio = boxW / float(boxH)
                    solidity = cv2.contourArea(c) / float(boxW * boxH)
                    heightRatio = boxH / float(filename4.shape[0])

                    # determine if the aspect ratio, solidity, and height of the contour pass
                    # the rules tests
                    keepAspectRatio = aspectRatio < 1.0
                    keepSolidity = solidity > 0.15
                    keepHeight = heightRatio > 0.4 and heightRatio < 0.95

                    # check to see if the component passes all the tests
                    #if keepAspectRatio and keepSolidity and keepHeight:
                        # compute the convex hull of the contour and draw it on the character
                        # candidates mask
                     #   hull = cv2.convexHull(c)
                     #   print(hull)
                      #  cv2.drawContours(charCandidates, [hull], -1, 255, -1)

        if x0 is not None and y0 is not None:
            for n, box in enumerate(boxes):
                x, y, w, h = box
                image_crop = filename6[y0+y-2 :y0+y + h+4, x0+x-1 :x0+x + w+2 ]
                segments[x] = image_crop

        segments_s = dict(sorted(segments.items(), key=lambda item: item[0], reverse=False))
        k = 0
        d = len(segments_s)
        #print("количество сегментов равно ", d)
        charCandidates1 = segmentation.clear_border(charCandidates)
        #print(charCandidates1[16])
        #cv2.imwrite("out4/segment-.jpg", charCandidates1)
        #cv2.imwrite("out4/fon.jpg", im33)
        if d > 7 and d < 10:
            a = []
            for key, val in segments_s.items():
                #print(segments_s[key])
                # print(type(segments_s[key]))
                if segments_s[key].size > 0:
                    segment0 = segments_s[key]

                    #print((segment0.shape))
                    ht1 = segment0.shape[0]
                    wd1 = segment0.shape[1]
                    #print(ht1,wd1)
                    if ht < ht1 or wd < wd1:
                        segment1 = cv2.resize(segment0, dim, interpolation=cv2.INTER_AREA)
                    else:
                        segment1 = cv2.resize(segment0, dim, interpolation=cv2.INTER_LINEAR)

                    #segment1 = resize(segments_s[key], (32, 32), anti_aliasing=True)
                    # print(segment1)
                    a.append(segment1)
                    #cv2.imwrite('seg/segment1-{0}.jpg'.format(k), segment1 * 255)
                    #k += 1


        else:
            a = []





    return filename4, d,filename5, filename6, a










