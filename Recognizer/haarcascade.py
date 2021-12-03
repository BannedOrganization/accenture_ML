import numpy as np
import winsound
import math
import time
import os
import json
import cv2
import findplate


def haarcascade(filename2, filename21):
    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
    ks = 3
    bt = 0
    image = filename2
    gray = cv2.GaussianBlur(image, (ks, ks), bt)
    image000 = None
    image001 = None
    #ch = -1
    #ch2 = 0
    #ch_n = 0
    pl_n3 = []
    y_c =[]
    plate = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(10, 80), maxSize=(600, 3000))
    #print(len(plate))
    #for (x, y, w, h) in plate:
    if len(plate) > 1:
        for i in range (0, len(plate)):
            y_c.append(plate[i][1])
        y_c_a = np.asarray(y_c)
        y_max_i = np.argmax(y_c_a)
        (x, y, w, h) = plate[y_max_i]

        image000 = image[y:y + h, x:x + w]
        #ch += 1
        #ch_n +=1
        #print("h=", h, "  w=", w, " x=", x, " y=", y)
        #print(y,"y")
        cv2.rectangle(filename21, (x, y), (x + w, y + h), (0, 0, 255), 3)

        image001 = filename21[y:y + h, x:x + w]
            #cv2.imwrite('out4/plate_color_c1.jpg', image001)
        result4 = image001
        result5 = filename21
        x0 = x
        y0 = y
    elif len(plate) !=0:
        (x, y, w, h) = plate[0]
        image000 = image[y:y + h, x:x + w]
        #print(y, "y")
        cv2.rectangle(filename21, (x, y), (x + w, y + h), (0, 0, 255), 3)
        image001 = filename21[y:y + h, x:x + w]
        result4 = image001
        result5 = filename21
        x0 = x
        y0 = y


    if image000 is not None:
        result3 = image001
        result4 = image001
        result5 = filename21

    else:
        result3 = image000
        result4 = filename21
        result5 = filename21
        x0 = None
        y0 = None

    return result3, result4, result5, x0, y0


