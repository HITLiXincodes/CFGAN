import os
import cv2
import math
import numpy as np
import torch
from skimage.feature import canny
from torch.utils.data import Dataset

sift = cv2.SIFT_create(contrastThreshold=0.03)

class SIFTDataset(Dataset):
    def __init__(self, num, file):
        self.num = num
        self.file = file
        self.filelist = sorted(os.listdir(self.file))

    def __getitem__(self, idx):
        return self.load_item(idx)

    def __len__(self):
        return self.num

    def load_item(self, idx):
        try:
            Ig = cv2.imread(self.file + self.filelist[idx])
            Ig = cv2.resize(Ig, (256, 256))
        except:
            print(idx)
            return 
        Si = self.load_sift(Ig)
        Ig = Ig.astype('float') / 127.5 - 1.
        return self.tensor(Ig), self.tensor(Si), self.filelist[idx]

    def load_sift(self, img):
        size = 256
        fealen = 128
        feature = np.zeros([size, size, fealen], dtype=float)
        result=np.zeros([128,128,fealen],dtype=float)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        meta, des = sift.detectAndCompute(np.uint8(img), None)
        if len(meta) == 0:
            return feature
        des = des.astype('float') / 127.5 - 1.
        used = []
        for i in range(len(meta)):
            a = int(math.ceil(meta[i].pt[1]) - 1)
            b = int(math.ceil(meta[i].pt[0])) - 1
            fea = list(des[i])
            if self.isEmpty(feature[a][b][:128]):
                feature[a][b][:128] = fea
                used.append(i)
            
        for i in range(0,size,2):
            for j in range(0,size,2):
                for p in range(i,i+2):
                    for q in range(j,j+2):
                        if p>=size or q >=size:
                            break
                        if feature[p][q].any()!=0:
                            result[math.ceil(p/2)][math.ceil(q/2)]=feature[p][q]
                            p=i+2
                            q=i+2
        return result

    def isEmpty(self, feature):
        for i in range(min(len(feature), 128)):
            if feature[i] != 0:
                return False
        return True

    def search_ab(self, feature, a, b, size=256):
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                ra = a + i
                rb = b + j
                if 0 <= ra <= size - 1 and 0 <= rb <= size - 1 and self.isEmpty(feature[ra][rb]):
                    return ra, rb
        return -1, -1

    def tensor(self, img):
        return torch.from_numpy(img).float().permute(2, 0, 1)

