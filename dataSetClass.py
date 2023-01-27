from torch.utils.data import Dataset, DataLoader
import numpy as np
from main import nodeCount, leafCount
import torch
import main

featureCount = leafCount
outputCount = -1
if main.nodeCount == 128:
    outputCount = 15
elif main.nodeCount == 1024:
    outputCount = 116
columns = featureCount + nodeCount - 1
print(featureCount, outputCount)

class MyDataset(Dataset):
    def __init__(self, fileName):
        print(fileName)
        self.dataset = np.memmap(fileName, dtype = 'float64', mode='r')
        self.dataset.shape = (-1, columns)
        self.dataset = np.float32(self.dataset)

        self.dataset = torch.from_numpy(self.dataset)
        self.X = self.dataset[:, :featureCount]
        self.y = self.dataset[:, featureCount:featureCount+outputCount]
    def __len__(self):
        return self.dataset.shape[0]
        #return 100000
    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        return X, y