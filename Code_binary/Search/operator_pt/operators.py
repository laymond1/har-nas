import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class ConvOp(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, dilation, groups=1, bias=True):
        super(ConvOp, self).__init__()
        padding = get_same_padding(kernel_size) # kernel 3->1, 5->2, 7->3, 9->4
        if isinstance(padding, int):
            padding *= dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias
            )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)

class DepthConvOp(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, dilation, groups=1, bias=False):
        super(ConvOp, self).__init__()
        padding = get_same_padding(kernel_size) # kernel 3->1, 5->2, 7->3, 9->4
        if isinstance(padding, int):
            padding *= dilation
        self.depthconv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=in_channels, bias=bias
            )
        self.pointconv = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, groups=groups, bias=bias
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.depthconv(x)
        x = self.pointconv(x)
        x = self.bn(x)
        return F.relu(x)

class Classifier(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0, activation=False):
        super(Classifier, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_features, out_features)
        if activation:
            self.act = nn.ReLU()

    def forward(self, x):
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.fc(x)
        if hasattr(self, 'act'):
            x = self.act(x)
        return x




class ConvBNReLU(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, dilation, groups=1, affine=True, activation=True):
    super(ConvBNReLU, self).__init__()
    padding = get_same_padding(kernel_size) # kernel 3->1, 5->2, 7->3, 9->4
    if isinstance(padding, int):
        padding *= dilation
    self.conv = nn.Conv1d(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias=False)
    self.bn = nn.BatchNorm1d(C_out, affine=affine)
    if activation:
      self.act = nn.ReLU()
    
  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    if hasattr(self, 'act'):
      x = self.act(x)
    return x