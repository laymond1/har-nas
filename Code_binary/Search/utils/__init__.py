# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import numpy as np
import torch
from torch.utils.data import TensorDataset

from utils.latency_estimator import *
from utils.my_modules import *
from utils.pytorch_utils import *


def make_divisible(v, divisor, min_val=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_val:
    :return:
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 1, 'invalid kernel size: %s' % kernel_size
        p = get_same_padding(kernel_size[0])
        return p
    assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2


def get_split_list(in_dim, child_num):
    in_dim_list = [in_dim // child_num] * child_num
    for _i in range(in_dim % child_num):
        in_dim_list[_i] += 1
    return in_dim_list

def split_layer(total_channels, num_groups):
    split = [int(np.ceil(total_channels / num_groups)) for _ in range(num_groups)]
    split[num_groups - 1] += total_channels - sum(split)
    return split

def list_sum(x):
    if len(x) == 1:
        return x[0]
    else:
        return x[0] + list_sum(x[1:])


def delta_ij(i, j):
    if i == j:
        return 1
    else:
        return 0

def load_dataset(path):
    Xtrain = np.load(path + 'train_x.npy')
    Xtest = np.load(path + 'test_x.npy')
    Ytrain = np.load(path + 'train_y.npy')
    Ytest = np.load(path + 'test_y.npy')

    train_dataset = TensorDataset(torch.tensor(Xtrain).float(), 
                                    torch.tensor(Ytrain).long())
    test_dataset = TensorDataset(torch.tensor(Xtest).float(), 
                                    torch.tensor(Ytest).long())
    return train_dataset, test_dataset

# path = 'Code/Search/dataset/ucihar/'
# x, test = load_dataset(path)
# # x.shape
# x[0][0].shape
# data = torch.utils.data.DataLoader(x)
# for batch, label in data:
#     print(batch.type())
#     break