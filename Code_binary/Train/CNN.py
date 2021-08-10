from sys import modules
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F

class BasicConv1d(nn.Module):
    def __init__(self, in_c, out_c, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_c, out_c, **kwargs)
        self.bn = nn.BatchNorm1d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x) 
        return F.relu_(x)

input = torch.randn(1, 3, 128)
net = CNNModel(3, 6, 128)
net(input)

class CNNModel(nn.Module):
    def __init__(self, in_c, class_num, segment_size):
        super(CNNModel, self).__init__()

        self.conv1 = BasicConv1d(in_c, 64, kernel_size=7, stride=1, padding=3)
        self.conv2 = BasicConv1d(64, 64, kernel_size=7, stride=1, padding=3)
        self.conv3 = BasicConv1d(64, 64, kernel_size=7, stride=1, padding=3)

        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=3, padding=1)
        length = math.floor((segment_size + 2) /3)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=3, padding=1)
        length = math.floor((length + 2) /3)
        self.maxpool3 = nn.MaxPool1d(kernel_size=3, stride=3, padding=1)
        length = math.floor((length + 2) /3)

        self.fc = nn.Sequential(
            nn.Linear(length*64, 512),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(512, class_num)
        )

    def forward(self, x): 
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)

        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits