import torch

from torch import nn
from collections import OrderedDict 


def conv_layer(c_in, c_out, ks=3, stride=1, relu=True, batchnorm=True):
    if batchnorm == True: bias = False
    else:                 bias = True
    layers = [
              nn.Conv1d(c_in, c_out, ks, padding=ks//2, stride=stride, bias=bias),
    ]
    if relu: layers.append(nn.ReLU())
    if batchnorm: layers.append(nn.BatchNorm1d(c_out, eps=1e-5, momentum=0.1))
    return nn.Sequential(*layers)


def linear_layer(c_in, c_out, prob=None, relu=True):
    layers = [
        nn.Linear(c_in, c_out),
    ]
    if prob: layers.append(nn.Dropout(prob))
    if relu: layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class Flatten(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, x): return x.view(x.size(0), -1)


class ActorNet(nn.Module):
    
    def __init__(self, cs, c_in=4, c_h=50, c_out=4, size=(24,)):

        super().__init__()
        layers = []
        layers.append(conv_layer(c_in, cs[0], ks=5, stride=1))
        for i in range(1, len(cs)):
            layers.append(conv_layer(cs[i-1], cs[i], stride=2))
        self.cnn = nn.Sequential(*layers)

        x_tmp = torch.rand(1, c_in, *size)
        with torch.no_grad():
            size = self.cnn(x_tmp).shape
            c_m = size[1] * size[2]
    
        layers = [
            Flatten(),
            linear_layer(c_m, c_h),
            linear_layer(c_h, c_out, relu=False),
        ]
        layers.append(nn.Tanh())
        self.fcn = nn.Sequential(*layers)

    def forward(self, x):
        x = self.cnn(x)
        return self.fcn(x)


class CriticNet(nn.Module):
    
    def __init__(self, cs_cnn, cs_fcn, c_out=1, size_state=(4, 24), size_action=(4,)):

        super().__init__()
        layers = []
        layers.append(conv_layer(size_state[0], cs_cnn[0], ks=5, stride=1))
        for i in range(1, len(cs_cnn)):
            layers.append(conv_layer(cs_cnn[i-1], cs_cnn[i], stride=2))
        layers.append(Flatten())
        self.cnn = nn.Sequential(*layers)

        x_tmp = torch.rand(1, *size_state)
        with torch.no_grad():
            c_h = self.cnn(x_tmp).size(1)
        
        layers = []
        layers.append(linear_layer(size_action[0] + c_h, cs_fcn[0]))
        for i in range(1, len(cs_fcn)):
            layers.append(linear_layer(cs_fcn[i-1], cs_fcn[i]))
        layers.append(linear_layer(cs_fcn[i], c_out, relu=False))
        self.fcn = nn.Sequential(*layers)

    def forward(self, state, action):
        x = self.cnn(state)
        return self.fcn(torch.cat([x, action], dim=1))
