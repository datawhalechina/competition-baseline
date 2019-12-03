import cv2
import os, glob, shutil, codecs, json
from tqdm import tqdm, tqdm_notebook
# %pylab inline



desc = {}
desc['abc'] = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

desc['train'] = []
desc['test'] = []
desc['pb'] = []

import pandas as pd
df_train_label = pd.read_csv('../input/train_id_label.csv')
df_submit = pd.read_csv('./crnn-pytorch/pb_rcnn_label.csv')
df_submit['label'] = df_submit['label'].apply(lambda x: ' '+x)
df_submit.columns = ['name', ' label']

df_train_label = pd.concat([df_train_label, df_submit], axis=0, ignore_index=True)
print(df_train_label.shape)

train_guanzi = df_train_label[' label'].apply(lambda x: x[-4:]).unique()


def checkImageIsValid(imagePath):
    img = cv2.imread(imagePath)
    if img is None:
        return False
    
    with open(imagePath, 'rb') as f:
        imageBin = f.read()
    
    if imageBin is None:
        return False
    
    try:
        imageBuf = np.fromstring(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        imgH, imgW = img.shape[0], img.shape[1]
        if imgH * imgW == 0:
            return False
        return True
    except:
        return False

bad_img_path = []
for x in df_train_label['name'].values:
    if not checkImageIsValid('./data/data/'+x):
        bad_img_path.append(x)


import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
X = np.zeros((df_train_label['name'].shape[0], 2))
kf = KFold(n_splits=24)
kf.get_n_splits(X)

print(kf)
fold_idx=0
for train_index, test_index in kf.split(X, df_train_label[' label'].apply(lambda x:x[1:2])):
    print("TRAIN:", train_index, "TEST:", test_index)
    
    desc['fold'+str(fold_idx)+'_train'] = []
    desc['fold'+str(fold_idx)+'_test'] = []
    
    for row in df_train_label.iloc[train_index].iterrows():
#         desc['fold'+str(fold_idx)+'_train'].append({'text':row[1][' label'].strip(), 'name':row[1]['name']})
#         continue

        if row[1]['name'] in bad_img_path:
            continue
            
        if checkImageIsValid('./data/data/'+row[1]['name']):
            desc['fold'+str(fold_idx)+'_train'].append({'text':row[1][' label'].strip(), 'name':row[1]['name']})
        else:
            print('./data/data/'+row[1]['name'])
            
    for row in df_train_label.iloc[test_index].iterrows():
#         desc['fold'+str(fold_idx)+'_test'].append({'text':row[1][' label'].strip(), 'name':row[1]['name']})
#         continue

        if row[1]['name'] in bad_img_path:
            continue

        if checkImageIsValid('./data/data/'+row[1]['name']):
            desc['fold'+str(fold_idx)+'_test'].append({'text':row[1][' label'].strip(), 'name':row[1]['name']})
        else:
            print('./data/data/'+row[1]['name'])
            
    fold_idx+=1

for row in glob.glob('../input/private_test_data/*'):
    desc['pb'].append({'text':'QJ69411105', 'name':row.split('/')[-1]})
    
with open('./data/desc.json', 'w') as outfile:
    json.dump(desc, outfile)