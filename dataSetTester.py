import numpy as np
import main
import dataSetClass
import matplotlib.pyplot as plt

for case in range(4):
    
    fileName = 'datasets/lung'+str(main.nodeCount)+'_continuous'+str(case)+'.npy'
    dataset = np.memmap(fileName, dtype = 'float64', mode='r')
    dataset.resize((int(dataset.shape[0]/dataSetClass.columns), dataSetClass.columns))
    
    volumes = dataset[:,:dataSetClass.featureCount]
    restrictions = dataset[:,dataSetClass.featureCount:]
    
    print("Dataset Shape: ", dataset.shape)
    print(np.max(volumes), " | ", np.min(volumes), " | ", np.mean(volumes))
    print(np.max(restrictions), " | ", np.min(restrictions), " | ", np.mean(restrictions))

    plt.style.use('_mpl-gallery')
    
    fig, ax = plt.subplots()
    VP = ax.boxplot(restrictions[:,np.subtract(main.oneEachDepth, 1)], positions=range(4), widths=0.7, patch_artist=True,
                    showmeans=True, showfliers=False)

    plt.show()