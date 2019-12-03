import random
import numpy as np
import cv2
import torch
import albumentations.augmentations.functional as albumentations

class ToTensor(object):
    def __call__(self, sample):
        sample["img"] = torch.from_numpy(sample["img"].transpose((2, 0, 1))).float()
#         sample["img"][0] = (sample["img"][0] - 0.485)/0.229
#         sample["img"][0] = (sample["img"][0] - 0.456)/0.224
#         sample["img"][0] = (sample["img"][0] - 0.406)/0.225
        
        sample["seq"] = torch.Tensor(sample["seq"]).int()
        return sample


class Resize(object):
    def __init__(self, size=(320, 32)):
        self.size = size

    def __call__(self, sample):
        if sample["img"] is None:
            return np.zeros((320, 32, 3))
            
        else:
            sample["img"] = cv2.resize(sample["img"], self.size)
            sample["img"] = sample["img"].astype(float)/255.0
            sample["img"][0] = (sample["img"][0] - 0.485)/0.229
            sample["img"][0] = (sample["img"][0] - 0.456)/0.224
            sample["img"][0] = (sample["img"][0] - 0.406)/0.225
            return sample


class Rotation(object):
    def __init__(self, angle=5, fill_value=0, p = 0.5):
        self.angle = angle
        self.fill_value = fill_value
        self.p = p

    def __call__(self, sample):
        if np.random.uniform(0.0, 1.0) < self.p:
            return sample
        h,w,_ = sample["img"].shape
        ang_rot = np.random.uniform(self.angle) - self.angle/2
        transform = cv2.getRotationMatrix2D((w/2, h/2), ang_rot, 1)
        sample["img"] = cv2.warpAffine(sample["img"], transform, (w,h), borderValue = self.fill_value)
        return sample


class Translation(object):
    def __init__(self, fill_value=0, p = 0.5):
        self.fill_value = fill_value
        self.p = p

    def __call__(self, sample):
        if np.random.uniform(0.0, 1.0) < self.p:
            return sample
        h,w,_ = sample["img"].shape
        trans_range = [w / 20, h / 20]
        tr_x = trans_range[0]*np.random.uniform()-trans_range[0]/2
        tr_y = trans_range[1]*np.random.uniform()-trans_range[1]/2
        transform = np.float32([[1,0, tr_x], [0,1, tr_y]])
        sample["img"] = cv2.warpAffine(sample["img"], transform, (w,h), borderValue = self.fill_value)
        return sample


class Scale(object):
    def __init__(self, scale=[0.5, 1.2], fill_value=0, p = 0.5):
        self.scale = scale
        self.fill_value = fill_value
        self.p = p

    def __call__(self, sample):
        if np.random.uniform(0.0, 1.0) < self.p:
            return sample
        h, w, _ = sample["img"].shape
        scale = np.random.uniform(self.scale[0], self.scale[1])
        transform = np.float32([[scale, 0, 0],[0, scale, 0]])
        sample["img"] = cv2.warpAffine(sample["img"], transform, (w,h), borderValue = self.fill_value)
        return sample

# add lyz
class Snow(object):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, sample):
        if np.random.uniform(0.0, 1.0) < self.p or not sample["aug"]:
            return sample
        h, w, _ = sample["img"].shape
        sample["img"] = albumentations.add_snow(sample["img"], snow_point=0.5, brightness_coeff=2)
        return sample
    
class Contrast(object):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, sample):
        if np.random.uniform(0.0, 1.0) < self.p:
            return sample
        h, w, _ = sample["img"].shape
        sample["img"] = albumentations.brightness_contrast_adjust(sample["img"], beta=np.random.uniform(0.0, 1.0)+0.1)
        # sample["img"] = cv2.GaussianBlur(sample["img"],(3,3),0)
        return sample
    
class Grid_distortion(object):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, sample):
        # print('grid', np.random.uniform(0.0, 1.0))
        
        if np.random.uniform(0.0, 1.0) < self.p:
            return sample
        h, w, _ = sample["img"].shape
        
        # grid_distortion
        if np.random.uniform(0.0, 1.0) < self.p:
            num_steps=15
            distort_limit=[-0.05,0.05]
            stepsx = [1 + random.uniform(distort_limit[0], distort_limit[1]) for i in
                              range(num_steps + 1)]
            stepsy = [1 + random.uniform(distort_limit[0], distort_limit[1]) for i in
                              range(num_steps + 1)]
            sample["img"]=albumentations.grid_distortion(sample["img"],5,stepsx, stepsy)
        # elastic_transform
        else:
            sample["img"]=albumentations.elastic_transform(sample["img"], alpha=5, sigma=1, alpha_affine=random.uniform(0,2), 
                                            interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101,)
        
        if np.random.uniform(0.0, 1.0) < self.p-0.2:
            sample["img"]=albumentations.jpeg_compression(sample["img"], random.randint(20, 100))
        return sample