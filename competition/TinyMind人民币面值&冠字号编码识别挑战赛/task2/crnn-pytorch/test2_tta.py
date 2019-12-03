import os
import cv2, glob, json
import string
from tqdm import tqdm
import click
import numpy as np
import pandas as pd
import random

from collections import Counter
from sklearn.externals import joblib

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset.test_data import TestDataset
from dataset.text_data import TextDataset
from dataset.collate_fn import text_collate
from dataset.data_transform import Resize, Rotation, Translation, Scale, Contrast, Snow, Grid_distortion
from models.model_loader import load_model
from torchvision.transforms import Compose

import editdistance

def pred_to_string(pred):
    seq = []
    for i in range(pred.shape[0]):
        label = np.argmax(pred[i])
        seq.append(label - 1)
    out = []
    for i in range(len(seq)):
        if len(out) == 0:
            if seq[i] != -1:
                out.append(seq[i])
        else:
            if seq[i] != -1 and seq[i] != seq[i - 1]:
                out.append(seq[i])
    out = ''.join('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'[i] for i in out)
    return out

def decode(pred):
    seq = []
    for i in range(pred.shape[0]):
        seq.append(pred_to_string(pred[i]))
    return seq

def test(net, data, abc, cuda, visualize, batch_size=256):
    data_loader = DataLoader(data, batch_size=batch_size, num_workers=4, shuffle=False, collate_fn=text_collate)

    count = 0.0
    tp = 0.0
    avg_ed = 0.0
    pred_pb = []
    iterator = tqdm(data_loader)
    for sample in iterator:
        imgs = Variable(sample["img"])
        if cuda:
            imgs = imgs.cuda()
        out = net(imgs, decode=True)
        gt = (sample["seq"].numpy() - 1).tolist()
        lens = sample["seq_len"].numpy().tolist()
        pos = 0
        key = ''
        for i in range(len(out)):
            gts = ''.join(abc[c] for c in gt[pos:pos+lens[i]])
            pos += lens[i]
            pred_pb.append(out[i])
            
            if gts == out[i]:
                tp += 1.0
            else:
                avg_ed += editdistance.eval(out[i], gts)
            count += 1.0
        if not visualize:
            iterator.set_description("acc: {0:.4f}; avg_ed: {0:.4f}".format(tp / count, avg_ed / count))

    acc = tp / count
    avg_ed = avg_ed / count
    return acc, avg_ed, pred_pb


def test_tta(net, data, abc, cuda, visualize, batch_size=128):
    pred_pb_tta = None
    
    for _ in range(7):
        data_loader = DataLoader(data, batch_size=batch_size, num_workers=10, shuffle=False, collate_fn=text_collate)
        iterator = tqdm(data_loader)
        
        pred_pb = []
        for sample in iterator:
            imgs = Variable(sample["img"])
            if cuda:
                imgs = imgs.cuda()
            out = net(imgs, decode=False)
            out = out.permute(1, 0, 2).cpu().data.numpy()
            
            pred_pb.append(out)
        
        if pred_pb_tta is None:
            pred_pb_tta = np.concatenate(pred_pb)
        else:
            pred_pb_tta += np.concatenate(pred_pb)
    return 0, 0, decode(pred_pb_tta)

@click.command()
@click.option('--data-path', type=str, default=None, help='Path to dataset')
@click.option('--abc', type=str, default='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', help='Alphabet')
@click.option('--seq-proj', type=str, default="20x40", help='Projection of sequence')
@click.option('--snapshot', type=str, default=None, help='Pre-trained weights')
@click.option('--backend', type=str, default="resnet18", help='Backend network')
@click.option('--input-size', type=str, default="320x80", help='Input size')
@click.option('--gpu', type=str, default='1', help='List of GPUs for parallel training, e.g. 0,1,2,3')
@click.option('--visualize', type=bool, default=False, help='Visualize output')

def main(data_path, abc, seq_proj, backend, snapshot, input_size, gpu, visualize):
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    cuda = True if gpu is not '' else False

    input_size = [int(x) for x in input_size.split('x')]
    seq_proj = [int(x) for x in seq_proj.split('x')]
    
    print(list(glob.glob('./tmp/fold*_best') + glob.glob('./tmp2/fold*_best')))
    fold_pred_pb_tta = []
    # for snapshot in glob.glob('./tmp/fold*_best')[:]:
    
    for snapshot in list(glob.glob('./tmp/fold*_best') + glob.glob('./tmp2/fold*_best'))[:]:
    
#     for snapshot in ['./tmp/fold12_train_crnn_resnet18_0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_0.997181964573',
#                     './tmp/fold13_train_crnn_resnet18_0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_0.995571658615',
#                     './tmp/fold3_train_crnn_resnet18_0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_0.993961352657',
#                     './tmp/fold5_train_crnn_resnet18_0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_0.994363929147']:
        if np.random.uniform(0.0, 1.0) < 1:
            transform = Compose([
                # Rotation(),
                Translation(),
                # Scale(),
                Contrast(),
                # Grid_distortion(),
                Resize(size=(input_size[0], input_size[1]))
            ])
        else:
            transform = Compose([
                # Rotation(),
                Translation(),
                # Scale(),
                Contrast(),
                # Grid_distortion(),
                Resize(size=(input_size[0], input_size[1]))
            ])
            
        if data_path is not None:
            data = TextDataset(data_path=data_path, mode="pb", transform=transform)
        else:
            data = TestDataset(transform=transform, abc=abc)
        print(snapshot)
        
        net = load_model(data.get_abc(), seq_proj, backend, snapshot, cuda).eval()
        acc, avg_ed, pred_pb = test_tta(net, data, data.get_abc(), cuda, visualize)
        fold_pred_pb_tta.append(pred_pb)
    
    with open('../data/desc.json') as up:
        data_json = json.load(up)
    
    fold_pred_pb = []
    if len(fold_pred_pb_tta) > 1:
        for test_idx in range(len(fold_pred_pb_tta[0])):
            test_idx_folds = [fold_pred_pb_tta[i][test_idx] for i in range(len(fold_pred_pb_tta))]

            test_idx_chars = []
            for char_idx in range(10):
                char_tta = [test_idx_folds[i][char_idx] for i in range(len(test_idx_folds)) 
                            if len(test_idx_folds[i]) > char_idx]
#                 if len(char_tta) < len(glob.glob('./tmp/fold*_best'))-2:
#                     print(test_idx, glob.glob('../../input/private_test_data/*')[test_idx])
                
                if len(char_tta) > 0:
                    char_tta = Counter(char_tta).most_common()[0][0]
                else:
                    char_tta = '*'
                    # print(test_idx, glob.glob('../../input/private_test_data/*')[test_idx])

                test_idx_chars += char_tta
            fold_pred_pb.append(''.join(test_idx_chars))
    
        joblib.dump(fold_pred_pb_tta, 'fold_tta.pkl')
        
        df_submit = pd.DataFrame()
        df_submit['name'] = [x['name'] for x in data_json['pb']]
        # print(fold_pred_pb_tta)
        df_submit['label'] = fold_pred_pb
    else:
        df_submit = pd.DataFrame()
        df_submit['name'] = [x['name'] for x in data_json['pb']]
        # print(fold_pred_pb_tta)
        df_submit['label'] = fold_pred_pb_tta[0]
    
    df_submit.to_csv('tmp_rcnn_tta10_pb.csv', index=None)
    print("Accuracy: {}".format(acc))
    print("Edit distance: {}".format(avg_ed))

# python test2_tta.py --snapshot tmp/crnn_resnet18_0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_best --visualize False --data-path ../data/
if __name__ == '__main__':
    main()
