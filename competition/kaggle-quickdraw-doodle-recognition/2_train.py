#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, codecs, glob
from PIL import Image, ImageDraw

import numpy as np
import pandas as pd
import cv2

import torch
torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled = False

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

import logging
logging.basicConfig(level=logging.DEBUG, filename='example.log',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d]: %(message)s')  # 

def draw_cv2(raw_strokes, size=256, lw=6, time_color=True):
    BASE_SIZE = 299
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for t, stroke in enumerate(eval(raw_strokes)):
        
        str_len = len(stroke[0])
        for i in range(len(stroke[0]) - 1):
            
            # dot dropout
            if np.random.uniform() > 0.95:
                continue
            
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i] + 22, stroke[1][i]  + 22),
                         (stroke[0][i + 1] + 22, stroke[1][i + 1] + 22), color, lw)
    
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size))
    else:
        return img

class QRDataset(Dataset):
    def __init__(self, img_drawing, img_label, img_size, transform=None):
        self.img_drawing = img_drawing
        self.img_label = img_label
        self.img_size = img_size
        self.transform = transform

    def __getitem__(self, index):
        img = np.zeros((self.img_size, self.img_size, 3))
        img[:, :, 0] = draw_cv2(self.img_drawing[index], self.img_size)
        img[:, :, 1] = img[:, :, 0]
        img[:, :, 2] = img[:, :, 0]
        img = Image.fromarray(np.uint8(img))
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = torch.from_numpy(np.array([self.img_label[index]]))
        return img, label

    def __len__(self):
        return len(self.img_drawing)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def get_resnet18():
    model = models.resnet18(True)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(512, 340)
    return model

def get_resnet34():
    model = models.resnet34(True)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(512, 340)
    return model

def get_resnet50():
    model = models.resnet50(True)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(2048, 340)
    return model

def get_resnet101():
    model = models.resnet101(True)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(2048, 340)
    return model

def main():
    df_train = pd.read_pickle(os.path.join('./data', 'train_' + dataset + '.pkl'))
    # df_train = df_train.reindex(np.random.permutation(df_train.index))
    df_val = pd.read_pickle(os.path.join('./data', 'val_' + dataset + '.pkl'))
    
    train_loader = torch.utils.data.DataLoader(
        QRDataset(df_train['drawing'].values, df_train['word'].values, imgsize,
                         transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            # transforms.RandomAffine(5, scale=[0.95, 1.05]),
                            transforms.ToTensor(),
                            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ),
        batch_size=1000, shuffle=True, num_workers=5,
    )

    val_loader = torch.utils.data.DataLoader(
        QRDataset(df_val['drawing'].values, df_val['word'].values, imgsize,
                         transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.ToTensor(),
                            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ),
        batch_size=1000, shuffle=False, num_workers=5,
    )
    
    if modelname == 'resnet18':
        model = get_resnet18()
    elif modelname == 'resnet34':
        model = get_resnet34()
    elif modelname == 'resnet50':
        model = get_resnet50()
    elif modelname == 'resnet101':
        model = get_resnet101()
    
    # model = nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load('./resnet50_64_7_0.pt'))
    # model.load_state_dict(torch.load('./resnet34_256_1_3280(82.7529_93.9964).pt'))
    
    model = model.cuda(0)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 3, 5, 7, 8], gamma=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader) / 10, gamma=0.95)
    
    print('Train:', df_train.shape[0], 'Val', df_val.shape[0])
    print('Epoch/Batch\t\tTrain: loss/Top1/Top3\t\tTest: loss/Top1/Top3')

    for epoch in range(50):
        train_losss, train_acc1s, train_acc5s = [], [], []
        for i, data in enumerate(train_loader):
            scheduler.step()
            model = model.train()
            train_img, train_label = data
            optimizer.zero_grad()
            
            # TODO: data paraell
            # train_img = Variable(train_img).cuda(async=True)
            # train_label = Variable(train_label.view(-1)).cuda()
            
            train_img = Variable(train_img).cuda(0)
            train_label = Variable(train_label.view(-1)).cuda(0)
            
            
            output = model(train_img)
            train_loss = loss_fn(output, train_label)
            
            train_loss.backward()
            optimizer.step()
            
            train_losss.append(train_loss.item())
            if i % 5 == 0:
                logging.info('{0}/{1}:\t{2}\t{3}.'.format(epoch, i, optimizer.param_groups[0]['lr'], train_losss[-1]))
            
            if i % int(len(train_loader) / 10) == 0:
                val_losss, val_acc1s, val_acc5s = [], [], []
                
                with torch.no_grad():
                    train_acc1, train_acc3 = accuracy(output, train_label, topk=(1, 3))
                    train_acc1s.append(train_acc1.item())
                    train_acc5s.append(train_acc3.item())
                
                    for data in val_loader:
                        val_images, val_labels = data
                        
                        # val_images = Variable(val_images).cuda(async=True)
                        # val_labels = Variable(val_labels.view(-1)).cuda()

                        val_images = Variable(val_images).cuda(0)
                        val_labels = Variable(val_labels.view(-1)).cuda(0) 
                       
                        output = model(val_images)
                        val_loss = loss_fn(output, val_labels)
                        val_acc1, val_acc3 = accuracy(output, val_labels, topk=(1, 3))
                        
                        val_losss.append(val_loss.item())
                        val_acc1s.append(val_acc1.item())
                        val_acc5s.append(val_acc3.item())
                        
                        if i == 0:
                            break
                
                logstr = '{0:2s}/{1:6s}\t\t{2:.4f}/{3:.4f}/{4:.4f}\t\t{5:.4f}/{6:.4f}/{7:.4f}'.format(
                    str(epoch), str(i),
                    np.mean(train_losss, 0), np.mean(train_acc1s, 0), np.mean(train_acc5s, 0),
                    np.mean(val_losss, 0), np.mean(val_acc1s, 0), np.mean(val_acc5s, 0),
                )
                torch.save(model.state_dict(), '{0}_{1}_{2}_{3}.pt'.format(modelname, imgsize, epoch, i))
                print(logstr)
                
    
# python 2_train.py 模型 数量 图片尺寸
# python 2_train.py resnet18 5000 64
if __name__ == "__main__":
    modelname = str(sys.argv[1])
    dataset = str(sys.argv[2])
    imgsize = int(sys.argv[3])
    main()