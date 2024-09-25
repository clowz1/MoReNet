import os
import glob
import pickle

import cv2
import torch
import torch.utils.data
import torch.nn as nn
import torchvision.transforms as trans
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import *


class MoReDataset(torch.utils.data.Dataset):
    def __init__(self, path, input_size, input_viewpoint, is_train=False, with_name=False):
        self.input_size = input_size
        self.input_viewpoint = np.array(input_viewpoint)
        self.path = path
        self.with_name = with_name
        self.is_train = is_train

        if is_train:
            self.label = np.load(os.path.join(path, 'angle1.npy'), allow_pickle=True)  # os.path.join 为拼接路径
        else:
            # self.label = np.load(os.path.join(path, 'jo0int_test.npy'))
            self.label = np.load(os.path.join(path, 'angle1.npy'), allow_pickle=True)

        self.length = len(self.label)

    def __getitem__(self, index):
        tag = self.label[index]
        fname = tag[0]
        # critical
        target = tag[1:].astype(np.float32)[-22:]
        # 列出tag第二行到最后一行

        hand = cv2.imread(os.path.join(self.path, fname+'.png'), cv2.IMREAD_COLOR).astype(np.float32)


        assert (hand.shape[0] == hand.shape[1] == self.input_size)
        hand = torch.tensor(human.transpose(2, 0, 1))




        if self.with_name:
            return hand, target, fname
        else:
            return hand, target

    def __len__(self):
        return self.length


if __name__ == '__main__':
    pass
