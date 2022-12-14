from torch.utils.data import Dataset, DataLoader
import numpy as np
from main import nodeCount, leafCount
import torch

featureCount = leafCount
outputCount = nodeCount - 1
columns = featureCount + outputCount
swap = False
if swap: 
    featureCount, outputCount = outputCount, featureCount
print(featureCount, outputCount)

class MyDataset(Dataset):
    def __init__(self, fileName):
        print(fileName)
        self.dataset = np.memmap(fileName, dtype = 'float64', mode='r')
        self.dataset.resize((int(self.dataset.shape[0]/columns), columns))
        self.dataset = torch.FloatTensor(self.dataset)
        if swap:
            self.X = self.dataset[:, outputCount:]
            self.y = self.dataset[:, :outputCount]
        else:
            self.X = self.dataset[:, :featureCount]
            self.y = self.dataset[:, featureCount:]
    def __len__(self):
        return self.dataset.shape[0]
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return x, y