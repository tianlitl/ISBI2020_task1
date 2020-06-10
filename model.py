#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class BasicConv2d(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(oup, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        return x

class Inception(nn.Module):
    #192
    def __init__(self, inp, c1, c2, c3, c4, c5, c6):
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(inp, c1, kernel_size=1, stride=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(inp, c2, kernel_size=1, stride=1),
            BasicConv2d(c2, c3, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(inp, c4, kernel_size=1, stride=1),
            BasicConv2d(c4, c5, kernel_size=3, stride=1, padding=1),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True),
            BasicConv2d(inp, c6, kernel_size=1, stride=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class GoogleNetDR(nn.Module):
    def __init__(self,num_classes = 5, transform_input=False):
        super(GoogleNetDR, self).__init__()
        self.transform_input = transform_input
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0,dilation=1,ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1, stride=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc_ = nn.Linear(1024, num_classes)
    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc_(x)
        return x

def load_pretrain_param_googlenet(googlenetdr = GoogleNetDR, path='googlenet-1378be20.pth'):
    googlenetdr_dict = googlenetdr.state_dict()
    googlenet = models.googlenet(pretrained=False)
    googlenet.load_state_dict(torch.load(path))
    googlenet_dict = googlenet.state_dict()
    pretrained_dict = {k: v for k, v in googlenet_dict.items() if k in googlenetdr_dict}
    googlenetdr_dict.update(pretrained_dict)
    googlenetdr.load_state_dict(googlenetdr_dict)
    return googlenetdr

def pre_nn():
    Googlenet = models.googlenet(pretrained=False)
    return Googlenet


class AlexNetDR(nn.Module):
    def __init__(self, num_classes=5):
        super(AlexNetDR, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            #nn.BatchNorm2d(64, affine=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            #nn.BatchNorm2d(192, affine=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer6 = nn.Sequential(
            nn.Linear(256 * 15 * 15, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.layer7 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.fc8 = nn.Linear(4096, num_classes)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*15*15)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.fc8(x)
        return x
def load_pretrain_param_alexnet(alexnetdr = AlexNetDR(), path='alexnet-owt-4df8aa71.pth'):
    alexnetdr_dict = alexnetdr.state_dict()
    alexnet = models.alexnet(pretrained=False)
    alexnet.load_state_dict(torch.load(path))
    alexnet_dict = alexnet.state_dict()
    pretrained_dict = {k: v for k, v in alexnet_dict.items() if k in alexnetdr_dict}
    alexnetdr_dict.update(pretrained_dict)
    alexnetdr.load_state_dict(alexnetdr_dict)
    return alexnetdr

class Ensemble(nn.Module):
    """
    Ensemble two models: Multiple two models outputs with trainable weight matrix A1 and A2, then add them up.
    """
    def __init__(self, model1, model2, num_classes=5):
        super(Ensemble, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.A1 = nn.Parameter(torch.eye(num_classes)/1.0/num_classes, requires_grad=True)
        self.A2 = nn.Parameter(torch.eye(num_classes)/1.0/num_classes, requires_grad=True)
    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        x = x1.mm(self.A1) + x2.mm(self.A2)
        return x

