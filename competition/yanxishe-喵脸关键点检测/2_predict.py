# -*- coding: utf-8 -*-
import os, sys, glob, argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

import time, datetime
import pdb, traceback

import cv2
# import imagehash
from PIL import Image

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b4') 

import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

class QRDataset(Dataset):
    def __init__(self, img_path, transform=None):
        self.img_path = img_path
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
    
    def __getitem__(self, index):
        start_time = time.time()
        img = Image.open(self.img_path[index]).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        return img,torch.from_numpy(np.array(int('PNEUMONIA' in self.img_path[index])))
    
    def __len__(self):
        return len(self.img_path)

class VisitNet(nn.Module):
    def __init__(self):
        super(VisitNet, self).__init__()
                
#         model = models.resnet18(True)
#         model.avgpool = nn.AdaptiveAvgPool2d(1)
#         model.fc = nn.Linear(512, 2)
#         self.resnet = model
        
        model = EfficientNet.from_pretrained('efficientnet-b0') 
        model._fc = nn.Linear(1280, 18)
        self.resnet = model
        
    def forward(self, img):        
        out = self.resnet(img)
        return out

def predict(test_loader, model, tta=10):
    # switch to evaluate mode
    model.eval()
    
    test_pred_tta = None
    for _ in range(tta):
        test_pred = []
        with torch.no_grad():
            end = time.time()
            for i, (input, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
                input = input.cuda()
                target = target.cuda()

                # compute output
                output = model(input)
                output = output.data.cpu().numpy()

                test_pred.append(output)
        test_pred = np.vstack(test_pred)
    
        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred
    
    return test_pred_tta

test_jpg = ['../test/{0}.jpg'.format(x) for x in range(0, 9526)]
test_jpg = np.array(test_jpg)

test_pred = None
for model_path in ['./resnet18_fold4.pt']:
    
    test_loader = torch.utils.data.DataLoader(
        QRDataset(test_jpg,
                transforms.Compose([
                            transforms.Resize((512, 512)),
                            # transforms.RandomHorizontalFlip(),
                            # transforms.RandomVerticalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ), batch_size=20, shuffle=False, num_workers=10, pin_memory=True
    )
        
    
    model = VisitNet().cuda()
    model.load_state_dict(torch.load(model_path))
    # model = nn.DataParallel(model).cuda()
    if test_pred is None:
        test_pred = predict(test_loader, model, 1)
    else:
        test_pred += predict(test_loader, model, 1)
    
# test_csv = pd.DataFrame()
# test_csv[0] = list(range(0, 1047))
# test_csv[1] = np.argmax(test_pred, 1)
# test_csv.to_csv('tmp.csv', index=None, header=None)

test_pred = pd.DataFrame(test_pred)
test_pred.columns = ['left_eye_x', 'left_eye_y', 'right_eye_x', 'right_eye_y',
       'mouth_x', 'mouth_y', 'left_ear1_x', 'left_ear1_y', 'left_ear2_x',
       'left_ear2_y', 'left_ear3_x', 'left_ear3_y', 'right_ear1_x',
       'right_ear1_y', 'right_ear2_x', 'right_ear2_y', 'right_ear3_x',
       'right_ear3_y']
test_pred = test_pred.reset_index()

img_size = []
for idx in (range(9526)):
    img_size.append(cv2.imread('../test/{0}.jpg'.format(idx)).shape[:2])

img_size = np.vstack(img_size)
test_pred['height'] = img_size[:, 0]
test_pred['width'] = img_size[:, 1]

for col in test_pred.columns:
    if '_x' in col:
        test_pred[col]*=test_pred['width']
    elif '_y' in col:
        test_pred[col]*=test_pred['height']

test_pred.astype(int).iloc[:, :-2].to_csv('tmp.csv', index=None, header=None)