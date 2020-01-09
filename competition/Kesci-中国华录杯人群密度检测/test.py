import h5py
import PIL.Image as Image
import numpy as np
import os
import glob
import scipy
from image import *
from model import CANNet
import torch
from torch.autograd import Variable

from sklearn.metrics import mean_squared_error,mean_absolute_error

from torchvision import transforms


transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])

# the folder contains all the test images
img_folder='../A/'
img_paths=[]

for img_path in glob.glob(os.path.join(img_folder, '*')):
    img_paths.append(img_path)

model = CANNet()

model = model.cuda()

checkpoint = torch.load('part_B_pre.pth.tar')

model.load_state_dict(checkpoint['state_dict'])

model.eval()

pred= []
gt = []

# for i in xrange(len(img_paths)):
#     img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()
#     img = img.unsqueeze(0)
#     h,w = img.shape[2:4]
#     h_d = h/2
#     w_d = w/2
#     img_1 = Variable(img[:,:,:h_d,:w_d].cuda())
#     img_2 = Variable(img[:,:,:h_d,w_d:].cuda())
#     img_3 = Variable(img[:,:,h_d:,:w_d].cuda())
#     img_4 = Variable(img[:,:,h_d:,w_d:].cuda())
#     density_1 = model(img_1).data.cpu().numpy()
#     density_2 = model(img_2).data.cpu().numpy()
#     density_3 = model(img_3).data.cpu().numpy()
#     density_4 = model(img_4).data.cpu().numpy()

#     pure_name = os.path.splitext(os.path.basename(img_paths[i]))[0]
#     # gt_file = h5py.File(img_paths[i].replace('.jpg','.h5').replace('images','ground_truth'),'r')
#     # groundtruth = np.asarray(gt_file['density'])
#     pred_sum = density_1.sum()+density_2.sum()+density_3.sum()+density_4.sum()
#     pred.append(pred_sum)
#     # gt.append(np.sum(groundtruth))
#     print(img_paths[i], pred_sum)

for i in xrange(len(img_paths)):
    img = Image.open(img_paths[i])
    print('')
    print(img.size)
    if img.size[0] > 1200:
        img = img.resize((1024, int(img.size[1]*1024.0/img.size[0])))
#     elif img.size[1] < 350:
#         img = img.resize((1024, int(img.size[1]*1024.0/img.size[0])))
    print(img.size)
    
    img2 = transform(img.transpose(Image.FLIP_LEFT_RIGHT).convert('RGB')).cuda()
    img = transform(img.convert('RGB')).cuda()
    img2 = img2.unsqueeze(0)
    img = img.unsqueeze(0)
    h,w = img.shape[2:4]
    h_d = h/2
    w_d = w/2
    
    density_1 = model(img.cuda()).data.cpu().numpy()
    density_2 = model(img2.cuda()).data.cpu().numpy()
    
#     # img = img.unsqueeze(0)
#     h,w = img.shape[2:4]
#     h_d = h/2
#     w_d = w/2
#     img_1 = Variable(img[:,:,:h_d,:w_d].cuda())
#     img_2 = Variable(img[:,:,:h_d,w_d:].cuda())
#     img_3 = Variable(img[:,:,h_d:,:w_d].cuda())
#     img_4 = Variable(img[:,:,h_d:,w_d:].cuda())
#     density_3 = model(img_1).data.cpu().numpy()
#     density_4 = model(img_2).data.cpu().numpy()
#     density_5 = model(img_3).data.cpu().numpy()
#     density_6 = model(img_4).data.cpu().numpy()
    
    pure_name = os.path.splitext(os.path.basename(img_paths[i]))[0]
    # gt_file = h5py.File(img_paths[i].replace('.jpg','.h5').replace('images','ground_truth'),'r')
    # groundtruth = np.asarray(gt_file['density'])
    pred_sum = density_1.sum() + density_2.sum()
    pred.append(pred_sum/2)
    # gt.append(np.sum(groundtruth))
    print(img_paths[i], pred_sum)

import pandas as pd
df = pd.DataFrame()
df['file'] = [os.path.basename(x) for x in img_paths]
df['man_count'] = pred
df['man_count'] = df['man_count'].round()
df['man_count'] = df['man_count'].astype(int)
df.loc[df['man_count'] > 100, 'man_count'] = 100
df.loc[df['man_count'] < 0, 'man_count'] = 0
df.to_csv('../tmp2.csv', index=None)