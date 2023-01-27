import main
import os
import warnings
import time
import multiprocessing as mp
import logging
import numpy as np
import math
import dataSetClass
import resource
import time
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

for curr_cnt in np.arange(50, 1000, 50, dtype=int):
    for case in range(16):
        disease_count = 0
        low_rest = 0
        high_rest = 0
        min_rest = 0.1

        disease_count = pow(2, int(case/4))
        high_rest = 1 - 0.2 * int(case%4)
        low_rest = high_rest - 0.2 
        print("Case: ", case)
        print("disease_count: ", disease_count)
        print("high_rest: ", high_rest)
        print("low_rest: ", low_rest)

        def randomContinuousRestrictions(nodeCount, diseasedCount):
            ret = np.full(nodeCount-1, 1.0)
            rng = np.random.default_rng()
            diseasedNodes = []
            if main.nodeCount == 128:
                diseasedNodes = np.arange(2, 16)
            elif main.nodeCount == 1028:
                diseasedNodes = np.arange(16, 128)
            rng.shuffle(diseasedNodes)
            diseasedNodes = diseasedNodes[:diseasedCount]
            for i in range(2, main.nodeCount):
                if i in diseasedNodes:
                    ret[i-1]  = max(min_rest, ret[math.floor(i/2)-1] * np.random.uniform(low = low_rest, high = high_rest))
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
                restrictions = randomContinuousRestrictions(main.nodeCount, disease_count)
                restrictionsList.append(restrictions)

            volumesList = main.multiThreadGenericLung(auxG = main.G.copy(), restrictions = restrictionsList, threadCount = threadCount)
            for i in range(0, 5):
                to_check = np.random.randint(0, len(restrictionsList)-1)
                check(volumesList[to_check], restrictionsList[to_check])
            new_data_set = np.concatenate((volumesList, restrictionsList), axis = 1)
            
            fileName = 'datasets/lung'+str(main.nodeCount)+'_continuous'+str(case)+'.npy'
            alreadyExists = os.path.exists(fileName)

            if alreadyExists:
                dataset_aux = np.memmap(fileName, dtype = 'float64', mode = 'r')
                old_rows = int(dataset_aux.shape[0] / dataSetClass.columns)
                del dataset_aux
                dataset = np.memmap(fileName, dtype = 'float64', mode='r+', shape = (old_rows + rows, dataSetClass.columns))
                dataset[-rows:,:] = new_data_set[:,:]
                dataset.flush()
                print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            else:
                dataset = np.memmap(fileName, dtype = 'float64', mode='w+', shape = (rows, dataSetClass.columns))
                dataset[-rows:,:] = new_data_set[:,:]
                dataset.flush()
                print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        fileName = 'datasets/lung'+str(main.nodeCount)+'_continuous'+str(case)+'.npy'
        alreadyExists = os.path.exists(fileName)
        cnt = curr_cnt
        if alreadyExists:
            dataset_aux = np.memmap(fileName, dtype = 'float64', mode = 'r')
            old_rows = int(dataset_aux.shape[0] / dataSetClass.columns)
            del dataset_aux
            cnt = int(cnt - (old_rows / (8 * 125)))
        print(cnt)
        for i in range(cnt):
            computeRows(threadCount = 8, cyclesPerThread = 125)

    #global_rows = 1000
    #time_list = []

    #for i in range(1, 17):
    #    start_time = time.time()
    #    print("Threads: ", i, " | Cycles per Thread: ", int(global_rows/i))
    #    computeRows(threadCount = i, cyclesPerThread = int(global_rows/i))
    #    end_time = time.time()
    #    print(i, ' Execution time = %.6f seconds' % (end_time-start_time))
    #    time_list.append(end_time-start_time)

    #np.save("times_list.npy", time_list)
    #plt.plot(time_list)