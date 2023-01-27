import numpy as np
import main

columns = main.nodeCount-1+int(main.nodeCount/2)
featureCount = int(main.nodeCount/2)

for case in range(16):
    fileName = 'datasets/lung'+str(main.nodeCount)+'_continuous'+str(case)+'.npy'
    mean_fileName = 'datasets/mean_variance/lung'+str(main.nodeCount)+'_continuous'+str(case)+'_mean.npy'
    variance_fileName = 'datasets/mean_variance/lung'+str(main.nodeCount)+'_continuous'+str(case)+'_variance.npy'
    dataset = np.memmap(fileName, dtype = 'float64', mode='r')
    dataset.shape = (-1, columns)
    dataset = np.float32(dataset)
    volumes = dataset[:,:featureCount]

    means = np.mean(volumes, axis = 0)
    variances = np.sqrt(np.var(volumes, axis = 0))

    np.save(mean_fileName, means)
    np.save(variance_fileName, variances)

    print("DONE CASE:", case)
    #print(means)
    #print(variances)
    #for i in range(10):
    #    print(np.divide(np.subtract(volumes[i], means), variances))