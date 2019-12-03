import cv2
import os, glob, shutil, codecs

import mxnet as mx
from matplotlib import pyplot as plt
import gluoncv
from gluoncv import model_zoo, data, utils

net = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', ctx=mx.gpu(0), pretrained=False)
net.load_parameters('./faster_rcnn_resnet50_v1b_voc_0002_0.0519.params')
net.classes = ['zipcode']
net.collect_params().reset_ctx(ctx = mx.gpu(0))

# MXNET_CUDNN_AUTOTUNE_DEFAULT=0 python 2_predict_faster_rcnn.py

with codecs.open('./data/train_data_box.csv', 'w') as up:
    for path in glob.glob('../input/train_data/*.jpg'):
        orig_img_cv2 = cv2.imread(path)
        x, orig_img = data.transforms.presets.rcnn.load_test(path)
        x = x.as_in_context(mx.gpu(0))
        box_ids, scores, bboxes = net(x)
        bboxes = bboxes.asnumpy()[0][0].astype(int)
        
        y1, x1, y2, x2 = bboxes
        x1*=(orig_img_cv2.shape[0]*1.0/orig_img.shape[0])
        x2*=(orig_img_cv2.shape[0]*1.0/orig_img.shape[0])
        
        y1*=(orig_img_cv2.shape[1]*1.0/orig_img.shape[1])
        y2*=(orig_img_cv2.shape[1]*1.0/orig_img.shape[1])
        
        x1, x2 = int(x1), int(x2)
        y1, y2 = int(y1), int(y2)
        
        # x1-=10; x2+=10
        # y1-=10; y2+=10
        
        # plt.imshow(orig_img_cv2[int(x1):int(x2), int(y1):int(y2), :])
        cv2.imwrite('./data/data/'+path.split('/')[-1], orig_img_cv2[int(x1):int(x2), int(y1):int(y2)])
        up.write('{0},{1},{2},{3},{4}\n'.format(path, x1, y1, x2, y2))

with codecs.open('./data/public_test_data_box.csv', 'w') as up:
    for path in glob.glob('../input/public_test_data/*.jpg'):
        orig_img_cv2 = cv2.imread(path)
        x, orig_img = data.transforms.presets.rcnn.load_test(path)
        x = x.as_in_context(mx.gpu(0))
        box_ids, scores, bboxes = net(x)
        bboxes = bboxes.asnumpy()[0][0].astype(int)
        
        y1, x1, y2, x2 = bboxes
        x1*=(orig_img_cv2.shape[0]*1.0/orig_img.shape[0])
        x2*=(orig_img_cv2.shape[0]*1.0/orig_img.shape[0])
        
        y1*=(orig_img_cv2.shape[1]*1.0/orig_img.shape[1])
        y2*=(orig_img_cv2.shape[1]*1.0/orig_img.shape[1])
        
        x1, x2 = int(x1), int(x2)
        y1, y2 = int(y1), int(y2)
        
        #x1-=10; x2+=10
        # y1-=10; y2+=10
        
        # plt.imshow(orig_img_cv2[int(x1):int(x2), int(y1):int(y2), :])
        cv2.imwrite('./data/data/'+path.split('/')[-1], orig_img_cv2[int(x1):int(x2), int(y1):int(y2)])
        up.write('{0},{1},{2},{3},{4}\n'.format(path, x1, y1, x2, y2))
        
with codecs.open('./data/private_test_data_box.csv', 'w') as up:
    for path in glob.glob('../input/private_test_data/*.jpg'):
        orig_img_cv2 = cv2.imread(path)
        x, orig_img = data.transforms.presets.rcnn.load_test(path)
        x = x.as_in_context(mx.gpu(0))
        box_ids, scores, bboxes = net(x)
        bboxes = bboxes.asnumpy()[0][0].astype(int)
        
        y1, x1, y2, x2 = bboxes
        x1*=(orig_img_cv2.shape[0]*1.0/orig_img.shape[0])
        x2*=(orig_img_cv2.shape[0]*1.0/orig_img.shape[0])
        
        y1*=(orig_img_cv2.shape[1]*1.0/orig_img.shape[1])
        y2*=(orig_img_cv2.shape[1]*1.0/orig_img.shape[1])
        
        x1, x2 = int(x1), int(x2)
        y1, y2 = int(y1), int(y2)
        
        #x1-=10; x2+=10
        # y1-=10; y2+=10
        
        # plt.imshow(orig_img_cv2[int(x1):int(x2), int(y1):int(y2), :])
        cv2.imwrite('./data/data/'+path.split('/')[-1], orig_img_cv2[int(x1):int(x2), int(y1):int(y2)])
        up.write('{0},{1},{2},{3},{4}\n'.format(path, x1, y1, x2, y2))