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
logging.basicConfig(level=logging.DEBUG, filename='example.log',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d]: %(message)s')  # 

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
        
        model = models.resnet18(True)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.resnet = model
        
#         model_name = 'se_resnet50' # could be fbresnet152 or inceptionresnetv2
#         model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
#         model.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
#         model = torch.nn.Sequential(*(list(model.children())[:-1]))
        
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
                 
def accuracy(outputs, targets):
    with torch.no_grad():
        batch_size = outputs[0].size(0)
        
        output_idx = []
        for output in outputs:
            _, pred = output.topk(1, 1, True, True)
            # pred = pred
            pred = pred.t().flatten()
            output_idx.append(pred.data.cpu().numpy())
        
        output_idx = np.vstack(output_idx)
        targets = [x.data.cpu().numpy() for x in targets]
        targets = np.vstack(targets)
        return ((targets == output_idx).mean(0) == 1).mean(), (targets == output_idx).mean(0) != 1
    
def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    
    train_acc = []
    train_losss = []
    for input,target0,target1,target2,target3,target4,target5,target6,target7,target8,target9 in tqdm(train_loader):
        optimizer.zero_grad()
        
        input = input.cuda(non_blocking=True)
        target0 = target0.cuda(non_blocking=True)
        target1 = target1.cuda(non_blocking=True)
        target2 = target2.cuda(non_blocking=True)
        target3 = target3.cuda(non_blocking=True)
        target4 = target4.cuda(non_blocking=True)
        target5 = target5.cuda(non_blocking=True)
        target6 = target6.cuda(non_blocking=True)
        target7 = target7.cuda(non_blocking=True)
        target8 = target8.cuda(non_blocking=True)
        target9 = target9.cuda(non_blocking=True)

        # compute output
        output0,output1,output2,output3,output4,output5,output6,output7,output8,output9 = model(input)
        loss0 = criterion(output0, target0)
        loss1 = criterion(output1, target1)
        loss2 = criterion(output2, target2)
        loss3 = criterion(output3, target3)
        loss4 = criterion(output4, target4)
        loss5 = criterion(output5, target5)
        loss6 = criterion(output6, target6)
        loss7 = criterion(output7, target7)
        loss8 = criterion(output8, target8)
        loss9 = criterion(output9, target9)
            
        loss = (loss0+loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9)/10.0
#         loss = torch.max([])
        # measure accuracy and record loss
#         acc = accuracy([output0,output1,output2,output3,output4,output5,output6,output7,output8,output9], 
#                         [target0,target1,target2,target3,target4,target5,target6,target7,target8,target9])
        
        # print(acc)
        # status = "loss_mean: {}; ACC: {}".format(np.mean([acc0,acc1,acc2,acc3,acc4,acc5,acc6,acc7,acc8,acc9]), 
        #                                          loss.item())
        # iterator.set_description(status)
            
        
        loss.backward()
        optimizer.step()
        train_losss.append(loss.item())
        
    return np.mean(train_losss)

def validate(val_loader, model, criterion):
    model.eval()
    
    val_acc = []
    val_loss = []
    val_error_idx = []
    val_prob = []
    with torch.no_grad():
        for i, (input,target0,target1,target2,target3,target4,target5,target6,target7,target8,target9) in enumerate(val_loader):
            input = input.cuda(non_blocking=True)
            target0 = target0.cuda(non_blocking=True)
            target1 = target1.cuda(non_blocking=True)
            target2 = target2.cuda(non_blocking=True)
            target3 = target3.cuda(non_blocking=True)
            target4 = target4.cuda(non_blocking=True)
            target5 = target5.cuda(non_blocking=True)
            target6 = target6.cuda(non_blocking=True)
            target7 = target7.cuda(non_blocking=True)
            target8 = target8.cuda(non_blocking=True)
            target9 = target9.cuda(non_blocking=True)

            # compute output
            output0,output1,output2,output3,output4,output5,output6,output7,output8,output9 = model(input)
            loss0 = criterion(output0, target0)
            loss1 = criterion(output1, target1)
            loss2 = criterion(output2, target2)
            loss3 = criterion(output3, target3)
            loss4 = criterion(output4, target4)
            loss5 = criterion(output5, target5)
            loss6 = criterion(output6, target6)
            loss7 = criterion(output7, target7)
            loss8 = criterion(output8, target8)
            loss9 = criterion(output9, target9)
            
            loss = (loss0+loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9)/10.0
            # measure accuracy and record loss
            acc, error_idx = accuracy([output0,output1,output2,output3,output4,output5,output6,output7,output8,output9], 
                            [target0,target1,target2,target3,target4,target5,target6,target7,target8,target9])
            
            output_prob = None
            for output in [output0,output1,output2,output3,output4,output5,output6,output7,output8,output9]:
                if output_prob is None:
                    output_prob = np.exp(output.max(1)[0].data.cpu().numpy())
                else:
                    output_prob += np.exp(output.max(1)[0].data.cpu().numpy())
            output_prob /= 10    
            
            val_acc.append(acc)
            val_loss.append(loss.item())
            val_error_idx += list(error_idx)
            val_prob += list(output_prob)
        
        print(np.where(val_error_idx)[0], np.mean(val_error_idx))
        names = []
        
        for idx in np.where(val_error_idx)[0]:
            print(val_loader.dataset.img_json[idx]['name'], val_prob[idx])
            names.append(val_loader.dataset.img_json[idx]['name'])
        
        print(','.join(names))
        return np.mean(val_acc), np.mean(val_loss)
        # print('VAL', np.mean(val_acc), np.mean(val_loss))
        
        
def main():
    with open('../data/desc.json') as up:
        data_json = json.load(up)

    for fold_idx in range(15):
        train_mode = 'fold{0}_train'.format(fold_idx)
        val_mode = 'fold{0}_test'.format(fold_idx)

        train_loader = torch.utils.data.DataLoader(
            QRDataset(data_json[train_mode],
                    Compose([
                                # transforms.RandomAffine(5),
                                # transforms.ColorJitter(hue=.05, saturation=.05),
                                Resize(80, 320),
                                # GridDistortion(p=.5, distort_limit=0.15,num_steps=5),
                                RandomBrightnessContrast(),
                                ElasticTransform(alpha=0.1, sigma=5, alpha_affine=2,),
                                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                ToTensor(),
                ])
            ), batch_size=100, shuffle=True, num_workers=20, pin_memory=True
        )

        val_loader = torch.utils.data.DataLoader(
            QRDataset(data_json[val_mode],
                    Compose([
                                # transforms.RandomAffine(5),
                                # transforms.ColorJitter(hue=.05, saturation=.05),
                                Resize(80, 320),
                                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                ToTensor(),
                ])
            ), batch_size=70, shuffle=True, num_workers=20, pin_memory=True
        )

        model = RMB_Net()
        model = model.cuda()
        # model = nn.DataParallel(model).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.Adam(model.parameters(), 0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.85)

        best_val_acc = 0.0
        for epoch_idx in range(10):
            train_loss = train(train_loader, model, criterion, optimizer, epoch_idx)
            val_acc, val_loss = validate(val_loader, model, criterion)
            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join('tmp', train_mode+'_best.pt'))
            
            print('{0}: Train_{1}, Val_{2}/{3}, best_{4}'.format(epoch_idx, train_loss, val_loss, val_acc, best_val_acc))
            logging.info('{0}: Train_{1}, Val_{2}/{3}, best_{4}'.format(epoch_idx, train_loss, val_loss, val_acc, best_val_acc))
        # break
main()