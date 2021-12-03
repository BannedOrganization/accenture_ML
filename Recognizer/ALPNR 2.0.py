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
from cv2 import VideoWriter_fourcc
from imutils.video import VideoStream
from imutils.video import FPS
from keras.models import load_model
from scipy.spatial import distance
from collections import deque
from collections import Counter
import json

import f_contour
import findplate
import haarcascade
import rot
import rot_image
import treshhold_ots
import nn_recognition2
import segmentation_final
import segmentation4

def Recognizer(path):
    model = load_model('letter_.h5')
    model2 = load_model('digit_.h5')

    dictionary_l = {0: 'A', 1: 'B', 2: 'C', 3: 'E', 4: 'H', 5: 'K', 6: 'M', 7: 'O', 8: 'P', 9: 'T', 10: 'X', 11: 'Y'}
    dictionary_d = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}

    vs = cv2.VideoCapture(path)

    conf = 0.7
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    frame_width = int(vs.get(3))
    frame_height = int(vs.get(4))

    cadr = 0
    r = 0
    k = 0
    cn = 0
    q = deque(maxlen=3)
    while True:

        (grabbed, frame) = vs.read()
        if not grabbed:
            break
        cadr += 1
        f0, f01 = treshhold_ots.threshold_ots(frame)
        f1, f11, f111, startX, startY = haarcascade.haarcascade(f0, f01)
        if f1 is not None:
            f2, f21, f22 = rot.rot(f1, f11, f111)

            # print("f2 есть")

            s_c = 0

            if f2 is not None:
                f5, s_c, f51, f52, p = segmentation4.segmentation_s(f2, f21, f22, startX, startY)

                if len(p) != 0:
                    f61, pl, tx, ty = nn_recognition2.recognition(p, f52, model, model2, dictionary_l, dictionary_d,
                                                                  startX, startY)
                    r += 1
                    # if pl == plate_t:
                    # k +=1
                    q.appendleft(pl)
                    # print(q)
                    count = Counter(q)
                    # print(count)
                    offten_pl = count.most_common(1)
                    # print((offten_pl[0])[0])
                    cv2.putText(f61, (offten_pl[0])[0], (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0),
                                lineType=cv2.LINE_AA)
                    # if (offten_pl[0])[0] == plate_t:
                    #    cn +=1


                else:
                    f61 = f52
            else:
                # print("Финал. Вывод изображения невозможен")
                f61 = frame
        else:
            # print("Финал. Классификаторы номеров на фото не обнаружили")
            f61 = frame
            # cv2.imshow("Frame", f61)

        # cv2.imshow("Frame", f61)

        # out.write(f61)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord("q"):
        #    value = {'plate_number' :(offten_pl[0])[0]}
        #    print(json.dumps(value))
        #    break
        # c +=1
    vs.release()
    cv2.destroyAllWindows()
    value = {'plate_number': (offten_pl[0])[0]}

    return json.dumps(value)

print(Recognizer('1.jpg'))