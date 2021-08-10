import sys
import os
import numpy as np
import torch
import torch.nn as nn

from Search.data_providers.ucihar import UCIHARDataProvider
from Search.operator_pt.operators import *
from Search.utils.pytorch_utils import *


class Zero(nn.Module):
    def forward(self, x):
        x = x
        return x

dataset = UCIHARDataProvider()
train_loader = dataset.train
test_loader = dataset.test

class Model1(nn.Module):
    def __init__(self):
        self.layer1 = ConvOp(6, 64, 7, 1, 2)
        self.layer2 = Zero()
        self.layer3 = Zero()
        self.maxpool = nn.MaxPool1d(3, 3, padding=1)
        self.classifier1 = Classifier(320, 512, 0.05)
        self.classifier2 = Classifier(512, 6, activation=True)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier1(x)
        x = self.classifier2(x)
        return x

class Model2(nn.Module):
    def __init__(self):
        self.layer1 = ConvOp(6, 64, 7, 1, 2)
        self.layer2 = ConvOp(6, 64, 7, 1, 2)
        self.layer3 = Zero()
        self.maxpool = nn.MaxPool1d(3, 3, padding=1)
        self.classifier1 = Classifier(320, 512, 0.05)
        self.classifier2 = Classifier(512, 6, activation=True)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier1(x)
        x = self.classifier2(x)
        return x

num_epoch = 100
net = Model1().to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
      net.parameters(),
      0.05)

max_f1 = 0
for epoch in range(num_epoch):
    
    # train
    train_loss, train_acc = AverageMeter(), AverageMeter()
    net.train()
    for batch, target in train_loader:
        batch, target = batch.to('cuda'), target.to('cuda')

        optimizer.zero_grad()
        logits = net(batch)
        loss = criterion(logits, target)

        loss.backward()
        optimizer.step()

        top1, _ = accuracy(logits, target, topk=(1,))
        train_loss.update(loss.item(), batch.size(0))
        train_acc.update(top1.item(), batch.size(0))

    print(f"Train Loss: {train_loss.avg}\t Train Acc: {train_acc.avg}")

    # test
    test_loss, test_acc = AverageMeter(), AverageMeter()
    net.eval()
    for batch, target in test_loader:
        batch, target = batch.to('cuda'), target.to('cuda')

        optimizer.zero_grad()
        logits = net(batch)
        loss = criterion(logits, target)

        loss.backward()
        optimizer.step()

        top1, _ = accuracy(logits, target, topk=(1,))
        test_loss.update(loss.item(), batch.size(0))
        test_acc.update(top1.item(), batch.size(0))
        
    print(f"Test Loss: {test_loss.avg}\t Test Acc: {test_acc.avg}")

    if max_f1 < test_acc:
        max_f1 = test_acc
        print(f"Best Acc : {test_acc.avg} at {epoch}")
    print(f"Best Acc : {test_acc.avg} still at {epoch}")
    