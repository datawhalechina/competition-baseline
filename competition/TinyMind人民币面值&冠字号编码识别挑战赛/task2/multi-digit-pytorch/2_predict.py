# -*- coding: utf-8 -*-
import os, sys, glob, argparse, json
import pandas as pd
import numpy as np
from tqdm import tqdm, tqdm_notebook
# import pretrainedmodels
import time, datetime
import pdb, traceback

import cv2
# import imagehash
from PIL import Image

from sklearn.model_selection import train_test_split, StratifiedKFold

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

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, Resize, Normalize,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose, ElasticTransform
)
from albumentations.pytorch import ToTensor

import logging
# logging.basicConfig(level=logging.DEBUG, filename='example.log',
#                     format='%(asctime)s - %(filename)s[line:%(lineno)d]: %(message)s')  # 

class QRDataset(Dataset):
    def __init__(self, img_json, transform=None):
        self.img_json = img_json
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
    
    def __getitem__(self, index):
        start_time = time.time()
        
        img = cv2.imread(os.path.join('../data/data/', self.img_json[index]['name']))        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        
        img_label_idx = self.img_json[index]['text'].strip()
        label0 = np.array(self.char2idx(img_label_idx[0]))
        label1 = np.array(self.char2idx(img_label_idx[1]))
        label2 = np.array(self.char2idx(img_label_idx[2]))
        label3 = np.array(self.char2idx(img_label_idx[3]))
        label4 = np.array(self.char2idx(img_label_idx[4]))
        label5 = np.array(self.char2idx(img_label_idx[5]))
        label6 = np.array(self.char2idx(img_label_idx[6]))
        label7 = np.array(self.char2idx(img_label_idx[7]))
        label8 = np.array(self.char2idx(img_label_idx[8]))
        label9 = np.array(self.char2idx(img_label_idx[9]))
        
        return img, torch.from_numpy(label0), torch.from_numpy(label1), \
                torch.from_numpy(label2), torch.from_numpy(label3), torch.from_numpy(label4),\
                torch.from_numpy(label5), torch.from_numpy(label6), torch.from_numpy(label7),\
                torch.from_numpy(label8), torch.from_numpy(label9)
    
    def __len__(self):
        return len(self.img_json)
    
    def char2idx(self, ch):
        return '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'.find(ch)
    
class RMB_Net(nn.Module):
    def __init__(self):
        super(RMB_Net, self).__init__()
        
        feat_size = 512
        self.fc0 = nn.Linear(feat_size, 36)
        self.fc1 = nn.Linear(feat_size, 36)
        self.fc2 = nn.Linear(feat_size, 36)
        self.fc3 = nn.Linear(feat_size, 36)
        self.fc4 = nn.Linear(feat_size, 36)
        self.fc5 = nn.Linear(feat_size, 36)
        self.fc6 = nn.Linear(feat_size, 36)
        self.fc7 = nn.Linear(feat_size, 36)
        self.fc8 = nn.Linear(feat_size, 36)
        self.fc9 = nn.Linear(feat_size, 36)
        
        model = models.resnet18(False)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.resnet = model
        
    def forward(self, img):
        feat = self.resnet(img)
        feat = feat.reshape(feat.size(0), -1)
        
        out0 = self.fc0(feat)
        out1 = self.fc1(feat)
        out2 = self.fc2(feat)
        out3 = self.fc3(feat)
        out4 = self.fc4(feat)
        out5 = self.fc5(feat)
        out6 = self.fc6(feat)
        out7 = self.fc7(feat)
        out8 = self.fc8(feat)
        out9 = self.fc9(feat)
        
        return F.log_softmax(out0, dim=1), F.log_softmax(out1, dim=1), F.log_softmax(out2, dim=1), \
                F.log_softmax(out3, dim=1), F.log_softmax(out4, dim=1), F.log_softmax(out5, dim=1), \
                F.log_softmax(out6, dim=1), F.log_softmax(out7, dim=1), F.log_softmax(out8, dim=1), \
                 F.log_softmax(out9, dim=1)

def predict(test_loader, model, tta=1):
    model.eval()
    
    val_acc = []
    val_loss = []
    
    predict_ttas = None
    with torch.no_grad():
        for _ in range(tta):
            predict_tta = []
            for i, (input,target0,target1,target2,target3,target4,target5,target6,target7,target8,target9) in enumerate(test_loader):
                input = input.cuda(non_blocking=True)

                # compute output
                output0,output1,output2,output3,output4,output5,output6,output7,output8,output9 = model(input)
                output0 = output0.data.cpu().numpy()
                output1 = output1.data.cpu().numpy()
                output2 = output2.data.cpu().numpy()
                output3 = output3.data.cpu().numpy()
                output4 = output4.data.cpu().numpy()
                output5 = output5.data.cpu().numpy()
                output6 = output6.data.cpu().numpy()
                output7 = output7.data.cpu().numpy()
                output8 = output8.data.cpu().numpy()
                output9 = output9.data.cpu().numpy()

                output = np.array([output0,output1,output2,output3,output4,output5,output6,
                               output7,output8,output9])
                predict_tta.append(output)
                # print(output.shape, output9.shape)
            predict_tta = np.concatenate(predict_tta, 1)
            # return predict_tta
        
            if predict_ttas is None:
                predict_ttas = predict_tta
            else:
                predict_ttas += predict_tta
    return predict_ttas/tta

def main():
    with open('../data/desc.json') as up:
        data_json = json.load(up)
    
    test_loader = torch.utils.data.DataLoader(
        QRDataset(data_json['pb'],
                Compose([
                                Resize(80, 320),
                                # GridDistortion(p=.5, distort_limit=0.15,num_steps=5),
                                RandomBrightnessContrast(),
                                ElasticTransform(alpha=0.1, sigma=5, alpha_affine=2,),
                                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                ToTensor(),
            ])
        ), batch_size=70, shuffle=False, num_workers=20, pin_memory=True
    )
    
    test_tta = None
    for model_path in glob.glob('tmp/*_best.pt')[:]:   
        print(model_path)
        
        model = RMB_Net()
        model = model.cuda()
        # model = nn.DataParallel(model).cuda()
        model.load_state_dict(torch.load(model_path))
        
        model_pred = predict(test_loader, model, 1)
        if test_tta is None:
            test_tta = model_pred
        else:
            test_tta += model_pred
     
    submit_lbls = []
    for idx in range(test_tta.shape[1]):
        idx_chars = ['0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'[np.argmax(test_tta[idx_char, idx, :])] 
                     for idx_char in range(10)]
        idx_chars = ''.join(idx_chars)
        submit_lbls.append(idx_chars)
        
    df = pd.DataFrame()
    df['name'] = [x['name'] for x in data_json['pb']]
    df['label'] = submit_lbls
    df.to_csv('tmp_rcnn_tta10_cnn.csv', index=None)
    
main()
