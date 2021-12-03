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




def threshold_ots(filename1):
    if filename1 is not None:
        img00 = filename1
        img2gray = cv2.cvtColor(img00, cv2.COLOR_BGR2GRAY)
        ret, img_bin = cv2.threshold(img2gray, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #cv2.imwrite("out4/ots.jpg", final_img)
        #result1 = final_img
        return img_bin, img00
    else:
        #print("видеопоток закончился")
        pass