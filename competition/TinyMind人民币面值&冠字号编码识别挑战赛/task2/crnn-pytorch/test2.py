import os
import cv2, glob
import string
from tqdm import tqdm
import click
import numpy as np
import pandas as pd

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


def test_tta(net, data, abc, cuda, visualize, batch_size=256):
    pred_pb_tta = None
    
    for _ in range(10):
        data_loader = DataLoader(data, batch_size=batch_size, num_workers=4, shuffle=False, collate_fn=text_collate)
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
@click.option('--abc', type=str, default=string.digits+string.ascii_uppercase, help='Alphabet')
@click.option('--seq-proj', type=str, default="10x20", help='Projection of sequence')
@click.option('--backend', type=str, default="resnet34", help='Backend network')
@click.option('--snapshot', type=str, default=None, help='Pre-trained weights')
@click.option('--input-size', type=str, default="320x32", help='Input size')
@click.option('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
@click.option('--visualize', type=bool, default=False, help='Visualize output')
def main(data_path, abc, seq_proj, backend, snapshot, input_size, gpu, visualize):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    cuda = True if gpu is not '' else False

    input_size = [int(x) for x in input_size.split('x')]
    transform = Compose([
        Rotation(),
        Translation(),
        # Scale(),
        Contrast(),
        Grid_distortion(),
        Resize(size=(input_size[0], input_size[1]))
    ])
    if data_path is not None:
        data = TextDataset(data_path=data_path, mode="pb", transform=transform)
    else:
        data = TestDataset(transform=transform, abc=abc)
    seq_proj = [int(x) for x in seq_proj.split('x')]
    net = load_model(data.get_abc(), seq_proj, backend, snapshot, cuda).eval()
    acc, avg_ed, pred_pb = test_tta(net, data, data.get_abc(), cuda, visualize)
    
    df_submit = pd.DataFrame()
    df_submit['name'] = [x.split('/')[-1] for x in glob.glob('../../input/public_test_data/*')]
    df_submit['label'] = pred_pb
    
    df_submit.to_csv('tmp_rcnn_tta10.csv', index=None)
    print("Accuracy: {}".format(acc))
    print("Edit distance: {}".format(avg_ed))

if __name__ == '__main__':
    main()
