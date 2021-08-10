import sys
import os
import numpy as np
import torch
import torch.nn as nn

from data_providers.ucihar import UCIHARDataProvider
from operator_pt.operators import *
from utils.pytorch_utils import *


class Zero(nn.Module):
    def forward(self, x):
        x = x
        return x

def weight_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        #nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        #truncated_normal_(m.weight, std=0.01)
        nn.init.trunc_normal_(m.weight, std=0.01)
        nn.init.constant_(m.bias, 0.01)


dataset = UCIHARDataProvider()
train_loader = dataset.train
test_loader = dataset.test

class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.layer1 = ConvOp(6, 64, 7, 1, 2)
        self.layer2 = Zero()
        self.layer3 = Zero()
        self.maxpool = nn.MaxPool1d(3, 3, padding=1)
        self.classifier1 = Classifier(320, 512, activation=True)
        self.classifier2 = Classifier(512, 6, dropout_rate=0.05)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier1(x)
        x = self.classifier2(x)
        return x

class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.layer1 = ConvOp(6, 64, 7, 1, 2)
        self.layer2 = ConvOp(64, 64, 7, 1, 2)
        self.layer3 = ConvOp(64, 64, 3, 1, 2)
        self.maxpool = nn.MaxPool1d(3, 3, padding=1)
        self.classifier1 = Classifier(320, 512, activation=True)
        self.classifier2 = Classifier(512, 6, dropout_rate=0.05)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier1(x)
        x = self.classifier2(x)
        return x

class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        self.layer1 = ConvOp(6, 64, 7, 1, 2)
        # self.layer2 = Zero()
        # self.layer3 = Zero()
        self.layer2 = ConvOp(64, 64, 7, 1, 2)
        self.layer3 = ConvOp(64, 64, 3, 1, 2)
        self.maxpool = nn.MaxPool1d(3, 3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier1 = Classifier(64, 512, activation=True)
        self.classifier2 = Classifier(512, 6, dropout_rate=0.05)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier1(x)
        x = self.classifier2(x)
        return x

num_epoch = 300
net = Model3().to('cuda')
net.apply(weight_init)
criterion = nn.CrossEntropyLoss().cuda()
f1_loss = F1_Loss(n_classes=6).cuda()
# optimizer = torch.optim.SGD(
#       net.parameters(),
#       0.001)
optimizer = torch.optim.Adam(
    net.parameters(),
    lr=5e-4,
    #betas=(0.5,0.9),
    weight_decay=5e-4
)
# Best Acc : 0.9165661878047007 still at 281
# Best Acc : 0.895288180406918 still at 293
max_f1 = 0

for epoch in range(num_epoch):
    
    # train
    train_loss, train_acc, train_f1 = AverageMeter(), AverageMeter(), AverageMeter()
    net.train()
    for batch, target in train_loader:
        batch, target = batch.to('cuda'), target.to('cuda')

        optimizer.zero_grad()
        logits = net(batch)
        loss = criterion(logits, target)

        loss.backward()
        optimizer.step()

        top1 = accuracy(logits, target, topk=(1,))[0]
        f1 = f1_loss(logits, target)
        train_loss.update(loss.item(), batch.size(0))
        train_acc.update(top1.item(), batch.size(0))
        train_f1.update(f1.item(), batch.size(0))

    print("Epoch[%d] Train Loss: %.4f\t Train Acc: %.2f\t Test F1-score: %.2f" % (epoch, train_loss.avg, train_acc.avg, train_f1.avg) )

    # test
    test_loss, test_acc, test_f1 = AverageMeter(), AverageMeter(), AverageMeter()
    net.eval()
    with torch.no_grad():
        for batch, target in test_loader:
            batch, target = batch.to('cuda'), target.to('cuda')

            logits = net(batch)
            loss = criterion(logits, target)

            top1 = accuracy(logits, target, topk=(1,))[0]
            f1 = f1_loss(logits, target)
            test_loss.update(loss.item(), batch.size(0))
            test_acc.update(top1.item(), batch.size(0))
            test_f1.update(f1.item(), batch.size(0))
        
    print("Epoch[%d] Test Loss: %.4f\t Test Acc: %.2f\t Test F1-score: %.2f" % (epoch, test_loss.avg, test_acc.avg, test_f1.avg))

    if max_f1 < test_f1.avg:
        max_f1 = test_f1.avg
        mac_acc = test_acc.avg
        best_epoch = epoch
        print(f"Best Acc : {max_f1} at {best_epoch}")
    else:
        print(f"Best Acc : {max_f1} still at {best_epoch}")
    