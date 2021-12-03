import numpy as np
from numpy import asarray
import scipy.special
import winsound
import math
import matplotlib.pyplot as plt
import sklearn
import argparse
from sklearn import preprocessing
from keras.models import load_model
import time
import tensorflow as tf
from PIL import Image
import matplotlib.image as mpimg
import os
import json
import cv2
import scipy.misc
import PIL


#model = load_model('letter_recognition99-28062019-2.h5')
#model2 = load_model('digit_recognition99-19062019-2-vgg.h5')

def recognition(a, filename7, model, model2, dictionary_l, dictionary_d, st_x, st_y):

    dictionary = dictionary_l
    dictionary2 = dictionary_d
    l = []


#print(img_resized3)
    for i in range(len(a)):
        if i == 0 or i == 4 or i == 5:

            img_data2 = a[i].astype('uint8')
            img_data30 = cv2.cvtColor(img_data2, cv2.COLOR_BGR2GRAY)/255.0

            X1_1 = np.reshape(img_data30, [-1, 32, 32, 1])
            k = (model.predict(X1_1, batch_size=1, verbose=0))
            label = np.argmax(k)
            l.append(dictionary[label])

        elif i == 1 or i ==2 or i ==3 or i == 6 or i == 7 or i == 8:
            img_data21 = a[i].astype('uint8')
            img_data31 = cv2.cvtColor(img_data21, cv2.COLOR_BGR2GRAY)/255.0

            X1_2 = np.reshape(img_data31, [-1, 32, 32, 1])
            k2 = (model2.predict(X1_2, batch_size=1, verbose=0))

            label2 = np.argmax(k2)
            l.append(dictionary2[label2])

    #textx = round(filename7.shape[0]/2)
    #texty = round(filename7.shape[1]/2)
    textx = st_x -10
    texty = st_y -15

    plate = ''.join(map(str, l))

    #cv2.putText(filename7, plate, (textx, texty), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), lineType=cv2.LINE_AA)

    #with open('plate.csv','a') as f:
     #   f.write(''.join( plate))
      #  f.write("\n")
       # f.close()

    #print(l,"номер")
    #print(type(l))


    return filename7, plate, textx, texty

