import torch.nn as nn
import torch.nn.functional as F
import dataSetClass

featureCount = dataSetClass.featureCount
outputCount = dataSetClass.outputCount

generalWidth = 60
width0 = generalWidth
width1 = generalWidth
width2 = generalWidth

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.BN = nn.BatchNorm1d(featureCount)
        self.IN = nn.InstanceNorm1d(featureCount)
        self.fc1 = nn.Linear(featureCount, width0)
        self.fc2 = nn.Linear(width0, width1)
        self.fc3 = nn.Linear(width1, width2)
        self.fc4 = nn.Linear(width2, outputCount)

    def forward(self, x):
        #x = self.IN(x)
        x = self.BN(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        if dataSetClass.swap:
            x = (((F.tanh(x) + 1) * 0.5) * 0.4) + 0.001
        else:
            x = (((F.tanh(x) + 1) * 0.5) * 0.9) + 0.1
        return x