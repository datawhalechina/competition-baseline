import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import string
import random

class TestDataset(Dataset):
    def __init__(self,
                 epoch_len = 10000,
                 seq_len = 8,
                 transform=None,
                 abc=string.digits):
        super().__init__()
        self.abc = abc
        self.epoch_len = epoch_len
        self.seq_len = seq_len
        self.transform = transform

    def __len__(self):
        return self.epoch_len

    def get_abc(self):
        return self.abc

    def set_mode(self, mode='train'):
        return

    def generate_string(self):
        return ''.join(random.choice(self.abc) for _ in range(self.seq_len))

    def get_sample(self):
        h, w = 64, int(self.seq_len * 64 * 2.5)
        pw = int(w / self.seq_len)
        seq = []
        img = np.zeros((h, w), dtype=np.uint8)
        text = self.generate_string()
        for i in range(len(text)):
            c = text[i]
            seq.append(self.abc.find(c) + 1)
            hs, ws = 32, 32
            symb = np.zeros((hs, ws), dtype=np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(symb, str(c), (3, 30), font, 1.2, (255), 2, cv2.LINE_AA)
            # Rotation
            angle = 60
            ang_rot = np.random.uniform(angle) - angle/2
            transform = cv2.getRotationMatrix2D((ws/2, hs/2), ang_rot, 1)
            symb = cv2.warpAffine(symb, transform, (ws, hs), borderValue = 0)
            # Scale
            scale = np.random.uniform(0.7, 1.0)
            transform = np.float32([[scale, 0, 0],[0, scale, 0]])
            symb = cv2.warpAffine(symb, transform, (ws, hs), borderValue = 0)
            y = np.random.randint(hs, h)
            x = np.random.randint(i * pw, (i + 1) * pw - ws)
            img[y-hs:y, x:x+ws] = symb
        nw = int(w * 32 / h)
        img = cv2.resize(img, (nw, 32))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img, seq

    def __getitem__(self, idx):
        img, seq = self.get_sample()
        sample = {"img": img, "seq": seq, "seq_len": len(seq), "aug": True}
        if self.transform:
            sample = self.transform(sample)
        return sample
