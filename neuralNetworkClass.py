import torch.nn as nn
import torch.nn.functional as F
import dataSetClass

featureCount = dataSetClass.featureCount
outputCount = dataSetClass.outputCount

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.m = nn.BatchNorm1d(featureCount)
        self.fc1 = nn.Linear(featureCount, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, outputCount)

    def forward(self, x):
        x = self.m(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        if dataSetClass.swap:
            x = (((F.tanh(x) + 1) * 0.5) * 0.2) + 0.2
        else:
            x = (((F.tanh(x) + 1) * 0.5) * 0.9) + 0.1
        return x