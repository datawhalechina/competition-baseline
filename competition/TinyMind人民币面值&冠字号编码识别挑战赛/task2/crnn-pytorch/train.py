import os
import click
import string
import numpy as np
from tqdm import tqdm
from models.model_loader import load_model
from torchvision.transforms import Compose
from dataset.data_transform import Resize, Rotation, Translation, Scale, Contrast, Snow, Grid_distortion
from dataset.test_data import TestDataset
from dataset.text_data import TextDataset
from dataset.collate_fn import text_collate
from lr_policy import StepLR

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch import Tensor
from torch.utils.data import DataLoader
from warpctc_pytorch import CTCLoss

from test import test

import logging
logging.basicConfig(level=logging.DEBUG, filename='example.log',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d]: %(message)s')  # 


@click.command()
@click.option('--data-path', type=str, default=None, help='Path to dataset')
@click.option('--abc', type=str, default='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', help='Alphabet')
@click.option('--seq-proj', type=str, default="20x40", help='Projection of sequence')
@click.option('--backend', type=str, default="resnet18", help='Backend network')
@click.option('--snapshot', type=str, default=None, help='Pre-trained weights')
@click.option('--input-size', type=str, default="320x80", help='Input size')
@click.option('--base-lr', type=float, default=1*1e-3, help='Base learning rate')
@click.option('--step-size', type=int, default=1500, help='Step size')
@click.option('--max-iter', type=int, default=6000, help='Max iterations')
@click.option('--batch-size', type=int, default=100, help='Batch size')
@click.option('--output-dir', type=str, default=None, help='Path for snapshot')
@click.option('--test-epoch', type=int, default=1, help='Test epoch')
@click.option('--test-init', type=bool, default=False, help='Test initialization')
@click.option('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')

def main(data_path, abc, seq_proj, backend, snapshot, input_size, base_lr, step_size, max_iter, batch_size, output_dir, test_epoch, test_init, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    cuda = True if gpu is not '' else False

    input_size = [int(x) for x in input_size.split('x')]
    transform = Compose([
        Rotation(),
        Translation(),
        # Scale(),
        Contrast(),
        # Grid_distortion(),
        Resize(size=(input_size[0], input_size[1]))
    ])
    seq_proj = [int(x) for x in seq_proj.split('x')]
    
    for fold_idx in range(24):
        train_mode = 'fold{0}_train'.format(fold_idx)
        val_mode = 'fold{0}_test'.format(fold_idx)
        
        if data_path is not None:
            data = TextDataset(data_path=data_path, mode=train_mode, transform=transform)
        else:
            data = TestDataset(transform=transform, abc=abc)
        
        net = load_model(data.get_abc(), seq_proj, backend, snapshot, cuda)
        optimizer = optim.Adam(net.parameters(), lr = base_lr, weight_decay=0.0001)
        lr_scheduler = StepLR(optimizer, step_size=step_size)
        # lr_scheduler = StepLR(optimizer, step_size=len(data)/batch_size*2)
        loss_function = CTCLoss()
        
        print(fold_idx)
        # continue
        
        acc_best = 0
        epoch_count = 0
        for epoch_idx in range(15):
            data_loader = DataLoader(data, batch_size=batch_size, num_workers=10, shuffle=True, collate_fn=text_collate)
            loss_mean = []
            iterator = tqdm(data_loader)
            iter_count = 0
            for sample in iterator:
                # for multi-gpu support
                if sample["img"].size(0) % len(gpu.split(',')) != 0:
                    continue
                optimizer.zero_grad()
                imgs = Variable(sample["img"])
                labels = Variable(sample["seq"]).view(-1)
                label_lens = Variable(sample["seq_len"].int())
                if cuda:
                    imgs = imgs.cuda()
                preds = net(imgs).cpu()
                pred_lens = Variable(Tensor([preds.size(0)] * batch_size).int())
                loss = loss_function(preds, labels, pred_lens, label_lens) / batch_size
                loss.backward()
                # nn.utils.clip_grad_norm(net.parameters(), 10.0)
                loss_mean.append(loss.data[0])
                status = "{}/{}; lr: {}; loss_mean: {}; loss: {}".format(epoch_count, lr_scheduler.last_iter, lr_scheduler.get_lr(), np.mean(loss_mean), loss.data[0])
                iterator.set_description(status)
                optimizer.step()
                lr_scheduler.step()
                iter_count += 1
            
            if True:
                logging.info("Test phase")
                
                net = net.eval()
                
#                 train_acc, train_avg_ed, error_idx = test(net, data, data.get_abc(), cuda, visualize=False)
#                 if acc > 0.95:
#                     error_name = [data.config[data.mode][idx]["name"] for idx in error_idx]
#                     logging.info('Train: '+','.join(error_name))
#                 logging.info("acc: {}\tacc_best: {}; avg_ed: {}\n\n".format(train_acc, train_avg_ed))

                data.set_mode(val_mode)
                acc, avg_ed, error_idx = test(net, data, data.get_abc(), cuda, visualize=False)
                
                if acc > 0.95:
                    error_name = [data.config[data.mode][idx]["name"] for idx in error_idx]
                    logging.info('Val: '+','.join(error_name))
                
                
                
                net = net.train()
                data.set_mode(train_mode)
                
                if acc > acc_best:
                    if output_dir is not None:
                        torch.save(net.state_dict(), os.path.join(output_dir, train_mode+"_crnn_" + backend + "_" + str(data.get_abc()) + "_best"))
                    acc_best = acc
                
                if acc > 0.985:
                    if output_dir is not None:
                        torch.save(net.state_dict(), os.path.join(output_dir, train_mode+"_crnn_" + backend + "_" + str(data.get_abc()) + "_"+str(acc)))
                logging.info("train_acc: {}\t; avg_ed: {}\n\n".format(acc, acc_best, avg_ed))
                
                
            epoch_count += 1

# python train.py --test-init True --test-epoch 10 --output-dir tmp --data-path ../data/
# python test2.py --snapshot tmp/crnn_resnet18_0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_best --visualize False --data-path ../data/
if __name__ == '__main__':
    main()
