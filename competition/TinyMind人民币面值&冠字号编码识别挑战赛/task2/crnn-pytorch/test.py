import os
import cv2
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
from dataset.data_transform import Resize, Rotation, Translation, Scale
from models.model_loader import load_model
from torchvision.transforms import Compose

import editdistance

def test(net, data, abc, cuda, visualize, batch_size=10):
    data_loader = DataLoader(data, batch_size=10, num_workers=1, shuffle=False, collate_fn=text_collate)
    
    error_idx = []
    idx= 0
    count = 0.0
    tp = 0.0
    avg_ed = 0.0
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
            
            if gts != out[i]:
                # print(out[i], gts, imgs.shape)
                error_idx.append(int(count))
            if gts == out[i]:
                tp += 1.0
            else:
                avg_ed += editdistance.eval(out[i], gts)
            count += 1.0
        if not visualize:
            iterator.set_description("acc: {0:.4f}; avg_ed: {1:.4f}".format(tp / count, avg_ed / count))
        idx+=1
        
    acc = tp / count
    avg_ed = avg_ed / count
    return acc, avg_ed, error_idx

@click.command()
@click.option('--data-path', type=str, default=None, help='Path to dataset')
@click.option('--abc', type=str, default=string.digits+string.ascii_uppercase, help='Alphabet')
@click.option('--seq-proj', type=str, default="10x20", help='Projection of sequence')
@click.option('--backend', type=str, default="resnet18", help='Backend network')
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
        Resize(size=(input_size[0], input_size[1]))
    ])
    if data_path is not None:
        data = TextDataset(data_path=data_path, mode="test", transform=transform)
    else:
        data = TestDataset(transform=transform, abc=abc)
    seq_proj = [int(x) for x in seq_proj.split('x')]
    net = load_model(data.get_abc(), seq_proj, backend, snapshot, cuda).eval()
    acc, avg_ed = test(net, data, data.get_abc(), cuda, visualize)
    
    df_submit = pd.DataFrame()
    
    print("Accuracy: {}".format(acc))
    print("Edit distance: {}".format(avg_ed))

if __name__ == '__main__':
    main()
