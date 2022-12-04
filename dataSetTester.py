import numpy as np

fileName  = "lung16_continuous.npy"
dataset = np.memmap(fileName, dtype = 'float64', mode='r')
dataset.resize((int(dataset.shape[0]/23), 23))
print(dataset.shape)
print(dataset[0])
print(dataset[-1])
print(np.max(dataset[:,0]))
print(np.min(dataset[:,0]))
print(np.mean(dataset[:,0]))
print(np.max(dataset[:,-1]))
print(np.min(dataset[:,-1]))
print(np.mean(dataset[:,-1]))