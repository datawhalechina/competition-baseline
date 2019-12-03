from torch.utils.data import Dataset
import json
import os
import cv2

class TextDataset(Dataset):
    def __init__(self, data_path, mode="train", transform=None):
        super(Dataset, self).__init__()
        self.data_path = data_path
        self.mode = mode
        self.config = json.load(open(os.path.join(data_path, "desc.json")))
        self.transform = transform

    def abc_len(self):
        return len(self.config["abc"])

    def get_abc(self):
        return self.config["abc"]

    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        if self.mode == "test":
            return len(self.config[self.mode])
        return len(self.config[self.mode])

    def __getitem__(self, idx):
        
        name = self.config[self.mode][idx]["name"]
        text = self.config[self.mode][idx]["text"]

        img = cv2.imread(os.path.join(self.data_path, "data", name))
        # print(os.path.join(self.data_path, "data", name))
        # img = cv2.imread(os.path.join(self.data_path, name))
        seq = self.text_to_seq(text)
        sample = {"img": img, "seq": seq, "seq_len": len(seq), "aug": self.mode == "train"}
        if self.transform:
            # print('trans')
            sample = self.transform(sample)
        return sample

    def text_to_seq(self, text):
        seq = []
        for c in text:
            seq.append(self.config["abc"].find(c) + 1)
        return seq