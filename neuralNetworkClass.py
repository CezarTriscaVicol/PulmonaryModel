import torch.nn as nn
import torch.nn.functional as F
import torch
import dataSetClass
import numpy as np

featureCount = dataSetClass.featureCount
outputCount = dataSetClass.outputCount

generalWidth = 2000           #50
width0 = 500
width1 = 100
width2 = 100
width3 = 100

class Net(nn.Module):
    def __init__(self, means, stds):
        super(Net, self).__init__()
        #self.N = nn.InstanceNorm1d(featureCount, affine = False, momentum = 0.0001)
        #self.N = nn.BatchNorm1d(featureCount, affine = False)
        self.means = means
        self.stds = stds
        self.fc1 = nn.Linear(featureCount, width0)
        self.fc2 = nn.Linear(width0, width1)
        self.fc3 = nn.Linear(width1, width2)
        self.fc4 = nn.Linear(width2, width3)
        self.fc5 = nn.Linear(width3, outputCount)
        self.xelu = F.elu

    def forward(self, x):
        #x = self.N(x)
        x = torch.div(torch.subtract(x, self.means), self.stds)
        x = self.fc1(x)
        x = self.xelu(x)
        x = self.fc2(x)
        x = self.xelu(x)
        x = self.fc3(x)
        x = self.xelu(x)
        x = self.fc4(x)
        x = self.xelu(x)
        x = self.fc5(x)
        x = (((F.tanh(x) + 1) * 0.5) * 0.9) + 0.1
        return x