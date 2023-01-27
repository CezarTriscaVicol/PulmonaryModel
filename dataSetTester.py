import numpy as np
import main
import dataSetClass
import matplotlib.pyplot as plt
import torch

def check(volumes, restrictions):
    aux_volumes = main.computeGenericLung(auxG = main.G.copy(), restrictions = restrictions)
    if np.mean(np.abs(np.subtract(volumes, aux_volumes))) > 0.00001:
        print("Error")
        #print(volumes)
        #print(aux_volumes)
        #print(restrictions)
        exit(0)    

for case in [15]:
    
    fileName = 'datasets/lung'+str(main.nodeCount)+'_continuous'+str(case)+'.npy'
    dataset = np.memmap(fileName, dtype = 'float64', mode='r')
    dataset.shape = (-1, dataSetClass.columns)
    #dataset = torch.from_numpy(dataset)
    dataset = np.float32(dataset)

    #def computeGenericLung(auxG, restrictions = np.full(nodeCount-1, 1), printLung = False, multiThreading = False):
    print("TESTING LUNG")
    main.computeGenericLung(auxG = main.G.copy(), restrictions = dataset[np.random.randint(dataset.shape[0]), dataSetClass.featureCount:], printLung = True, multiThreading = False)

    volumes = dataset[:,:dataSetClass.featureCount]
    restrictions = dataset[:,dataSetClass.featureCount:]

    print(type(volumes[0][0]))
    print(type(restrictions[0][0]))
    
    print("Case: ", case)
    print("Dataset Shape: ", dataset.shape)
    print(np.max(volumes), " | ", np.min(volumes), " | ", np.mean(volumes))
    print(np.max(restrictions), " | ", np.min(restrictions), " | ", np.mean(restrictions))

    #for i in range(100):
    #    idx = np.random.randint(0, dataset.shape[0])
    #    check(volumes[idx], restrictions[idx])

    plt.style.use('_mpl-gallery')
    
    fig, ax = plt.subplots()
    VP = ax.boxplot(restrictions[:,np.subtract(main.oneEachDepth, 1)], positions=range(7), widths=0.7, patch_artist=True,
                    showmeans=True, showfliers=False)

    plt.show()