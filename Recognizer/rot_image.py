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


def rot_image(image, angler):
    if angler == 1000:
        #print("Поворот фото. Дальнейшие вычисления невозможны")
        result2 = None
    else:
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angler, 1.0)
        result2 = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result2