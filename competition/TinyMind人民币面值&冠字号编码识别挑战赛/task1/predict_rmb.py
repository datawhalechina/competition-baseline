# -*- coding: utf-8 -*-
import os, sys, glob, argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

import time, datetime
import pdb, traceback

import cv2
from PIL import Image

from sklearn.preprocessing import LabelEncoder
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

class QRDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label=img_label
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
    
    def __getitem__(self, index):
        start_time = time.time()
        img = Image.open(self.img_path[index])
        
        if self.transform is not None:
            img = self.transform(img)
                
        return img, torch.from_numpy(np.array([self.img_label[index]]))
    
    def __len__(self):
        return len(self.img_path)
        
class VisitNet(nn.Module):
    def __init__(self):
        super(VisitNet, self).__init__()
        model = models.resnet18(False)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(512, 256)
        self.resnet = model
        
    def forward(self, img):
        out = self.resnet(img)
        return F.log_softmax(out, dim=1)
    
def predict(test_loader, model, tta=10):
    # switch to evaluate mode
    model.eval()
    
    test_pred_tta = None
    for _ in range(tta):
        test_pred = []
        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(test_loader):
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


def main():
    
    # 修改输入的路径
    df_train = pd.read_csv('../../input/train_face_value_label.csv', dtype={' label': object, 'name': object})
    lbl = LabelEncoder()
    df_train['y'] = lbl.fit_transform(df_train[' label'].values)
    
    # 修改输入的路径
    test_path = glob.glob('../../input/public_test_data/*.jpg')
    test_path = np.array(test_path)
    
    test_loader = torch.utils.data.DataLoader(
        QRDataset(test_path, np.zeros(len(test_path)),
                transforms.Compose([
                            # transforms.Resize((124, 124)),
                            transforms.Resize(280),
                            transforms.RandomCrop((256, 256)),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ), batch_size=10, shuffle=False, num_workers=10, pin_memory=True
    )

    model = VisitNet()
    model = model.cuda()
    model.load_state_dict(torch.load('./resnet18_fold0_11_Acc@1100.00(100.00).pt'))
    
    test_pred = predict(test_loader, model, 10)
    test_pred = np.vstack(test_pred)
    test_pred = np.argmax(test_pred, 1)
    
    test_pred = lbl.inverse_transform(test_pred)
    test_csv = pd.DataFrame()
    test_csv['name'] = [x.split('/')[-1] for x in test_path]
    test_csv['label'] = test_pred
    test_csv.sort_values(by='name', inplace=True)
    test_csv.to_csv('tmp_newmodel_resnet18_tta10.csv', index=None, sep=',')

if __name__== "__main__":
    main()
