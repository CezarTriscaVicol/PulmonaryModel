import main
import warnings
import time
import multiprocessing as mp
import logging
import numpy as np
import math
import dataSetClass
import resource
warnings.filterwarnings("ignore")

case = 2
dataset_case = case + main.nodeCount

disease_prob = 0
low_rest = 0
high_rest = 0
min_rest = 0.1

if dataset_case == 16:
    disease_prob = 0.2
    low_rest = 0.3
    high_rest = 0.9

elif dataset_case == 17:
    disease_prob = 0.32
    low_rest = 0.4
    high_rest = 0.7

elif dataset_case == 18:
    disease_prob = 0.1
    low_rest = 0.4
    high_rest = 0.6

elif dataset_case == 19:
    disease_prob = 0.1
    low_rest = 0.2
    high_rest = 0.5

elif dataset_case == 1024:
    disease_prob = 0.05
    low_rest = 0.3
    high_rest = 0.9

def randomContinousRestricions(nodeCount):
    ret = np.full(nodeCount-1, 1.0)
    at_least_one_change = False
    while not at_least_one_change:
        for i in range(2, main.nodeCount):
            if np.random.uniform() < disease_prob:
                auxx = max(min_rest, ret[math.floor(i/2)-1] * np.random.uniform(low = low_rest, high = high_rest))
                ret[i-1] = auxx
                at_least_one_change = True
            else:
                ret[i-1] = ret[math.floor(i/2)-1]
    return ret
def check(volumes, restrictions):
    aux_volumes = main.computeGenericLung(auxG = main.G.copy(), restrictions = restrictions)
    if np.mean(np.abs(np.subtract(volumes, aux_volumes))) > 0.00001:
        print("Error")
        print(volumes)
        print(aux_volumes)
        print(restrictions)
        exit(0)    


def computeRows(threadCount = 8, cyclesPerThread = 0):
    rows = threadCount * cyclesPerThread

    restrictionsList = []

    for i in range(0, rows):
        restrictions = randomContinousRestricions(main.nodeCount)
        restrictionsList.append(restrictions)

    volumesList = main.multiThreadGenericLung(auxG = main.G.copy(), restrictions = restrictionsList, threadCount = threadCount)
    for i in range(0, 10):
        to_check = np.random.randint(0, len(restrictionsList)-1)
        check(volumesList[to_check], restrictionsList[to_check])
    new_data_set = np.concatenate((volumesList, restrictionsList), axis = 1)
    
    fileName = 'datasets/lung'+str(main.nodeCount)+'_continuous'+str(case)+'.npy'
    dataset_aux = np.memmap(fileName, dtype = 'float64', mode = 'r')
    old_rows = int(dataset_aux.shape[0] / dataSetClass.columns)
    del dataset_aux
    dataset = np.memmap(fileName, dtype = 'float64', mode='r+', shape = (old_rows + rows, dataSetClass.columns))
    #dataset = np.memmap(fileName, dtype = 'float64', mode='w+', shape = (rows, dataSetClass.columns))
    dataset[-rows:,:] = new_data_set[:,:]
    dataset.flush()
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

for i in range(1000000):
    computeRows(threadCount = 7, cyclesPerThread = 1000)