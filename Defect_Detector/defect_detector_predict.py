#import os
import cv2
#import time
import pandas as pd
import numpy as np

#from PIL import Image

import torch
import torchvision
import torchvision.transforms as T

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset

import json


class GarbageDetectionDataset(Dataset):
    
    def __init__(self, path, mode = 'test', transforms = None):
        
        super().__init__()

        self.image_name = path
        self.transforms = transforms
        self.mode = mode
        
        
    def __getitem__(self, index: int):
        
        image = cv2.imread(self.image_name, cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32')
        image /= 255.0
        
        if self.mode == 'train':
          pass

        
        elif self.mode == 'test':

            if self.transforms:
                image = self.transforms(image)

            
            return image, self.image_name
    
    def __len__(self):
        #return len(self.image_name)
        return 1


def get_transform():
    return T.Compose([T.ToTensor()])


def collate_fn(batch):
    return tuple(zip(*batch))


def Garbage_Detector(path_to_image):

    _classes = ['defect', 'carriage']

    class_to_int = {'defect': 1, 'carriage': 2}
    int_to_class = {1: 'defect', 2: 'carriage'}

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache()

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = len(class_to_int)+1
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    checkpoint = torch.load("model_03122021_20.pth", map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)

  
    submission = []

    test_dataset = GarbageDetectionDataset(path_to_image, mode = 'test', transforms = get_transform())

    test_data_loader = DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=1,drop_last=False,collate_fn=collate_fn)

    threshold = 0.6
    model.eval()

    image_open = cv2.imread(path_to_image, cv2.IMREAD_COLOR)

    total_sum_defect = 0
    s_carriage = 0

    for images, image_names in test_data_loader:



        images = list(image.to(device) for image in images)
        output = model(images)

        boxes = output[0]['boxes'].data.cpu().numpy()
        scores = output[0]['scores'].data.cpu().numpy()
        labels = output[0]['labels'].data.cpu().numpy()

        boxes_th = boxes[scores >= threshold].astype(np.int32)
        scores_th = scores[scores >= threshold]

        labels_th = []
        for x in range(len(labels)):
            if scores[x] > threshold:
                labels_th.append(int_to_class[labels[x]])

        for y in range(len(boxes_th)):
            x1 = boxes_th[y][0]
            y1 = boxes_th[y][1]
            x2 = boxes_th[y][2]
            y2 = boxes_th[y][3]
            class_name = labels_th[y]

            if class_name == 'carriage':
                w_carriage = x2 - x1
                h_carriage = y2 - y1
                s_carriage = w_carriage * h_carriage

            else:
                w_defect = x2 - x1
                h_defect = y2 - y1
                s_defect = w_defect * h_defect

                total_sum_defect = total_sum_defect + s_defect



            row = {"image_name": image_names[0], "xmin": x1, "xmax": x2, "ymin": y1, "ymax": y2, "type": class_name}

            submission.append(row)

            if len(boxes_th) > 0:
                cv2.rectangle(image_open, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image_open, class_name, (int(x1), int(y1 - 10)), font, 1, (255, 0, 0), 2)

    if len(submission) == 0:
        result = "Defect not detected"
    else:
        result = "Defect detected"

    if s_carriage != 0 and total_sum_defect != 0:
        defect_percents = round((total_sum_defect * 100) / s_carriage, 2)
    else:
        defect_percents = 0

    value = {'Cargo': result, 'Defect percents': defect_percents ,'Image': image_open.tolist()}

    return json.dumps(value)


