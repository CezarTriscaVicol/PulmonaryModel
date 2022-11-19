import main
import warnings
import time
import multiprocessing as mp
import logging
import numpy as np
import math
warnings.filterwarnings("ignore")

computeNewRestrictions = True

def randomRestrictoins():
    return np.random.uniform(low = 0.2, high = 1.0, size = main.nodeCount-1)

def randomContinousRestricions():
    ret = [1]
    for i in range(2, main.nodeCount):
        if np.random.uniform() < 0.05:
            auxx = max(0.01, ret[math.floor(i/2)-1] * np.random.uniform(low = 0.3, high = 0.9))
            ret.append(auxx)
        else:
            ret.append(ret[math.floor(i/2)-1])
    return ret

def restrictionsToNew(restrictions):
    newRestrictions = [restrictions[0]]
    for j in range(1,restrictions.shape[0]):
        newRestrictions.append(restrictions[(j+1)-1] / restrictions[math.floor((j+1)/2)-1])
    return newRestrictions

def newToRestrictions(newRestrictions):
    restrictions = [newRestrictions[0]]
    for j in range(1,newRestrictions.shape[0]):
        restrictions.append(newRestrictions[(j+1)-1] * restrictions[math.floor((j+1)/2)-1])
    return restrictions

def check(volumes, restrictions):
    aux_volumes = main.computeGenericLung(auxG = main.G.copy(), restrictions = restrictions)
    if np.mean(np.abs(np.subtract(volumes, aux_volumes))) > 0.00001:
        print("Error")
        print(volumes)
        print(aux_volumes)
        print(restrictions)
        exit(0)

def computeRows(threadCount = 8, cyclesPerThread = 0, cnt = 0):
    rows = threadCount * cyclesPerThread

    restrictionsList = []

    if computeNewRestrictions:
        for i in range(0, rows):
            restrictions = randomContinousRestricions()
            restrictionsList.append(restrictions)
    else:
        restrictionsList = np.load("restrictionsList"+str(main.nodeCount)+".npy")
        restrictionsList = restrictionsList[:rows]

    volumesList = main.multiThreadGenericLung(auxG = main.G.copy(), restrictions = restrictionsList, threadCount = threadCount)
    for i in range(0, 2):
        to_check = np.random.randint(0, len(restrictionsList)-1)
        check(volumesList[to_check], restrictionsList[to_check])
    data_set = np.concatenate((volumesList, restrictionsList), axis = 1)
    
    np.save('lung'+str(main.nodeCount)+'_continuous.npy', data_set)

#for i in range(100):
#    computeRows(threadCount = 8, cyclesPerThread = 250, cnt = i)