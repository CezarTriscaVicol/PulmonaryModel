import matplotlib.pyplot as plt
import numpy as np
import main
import dataSetClass

plt.rc('axes', labelsize=20) 
fig = plt.figure(0, figsize = (20, 20)) 
fig.tight_layout()
for case in range(16):
    disease_count = pow(2, int(case/4))
    high_rest = 1 - 0.2 * int(case%4)
    low_rest = high_rest - 0.2 
    current_fig = fig.add_subplot(4, 4, case+1)
    if case < 4:
        current_fig.set_title("High: "+str('%.1f' % high_rest)+" | Low: "+str('%.1f' % low_rest))
    current_fig.set_title(current_fig.get_title()+"\nCase: "+str(case), fontsize = 20)
    if case % 4 == 0:
        current_fig.set_ylabel("Diseased Count: "+str(disease_count) )
    max_diff_vector = np.load("graph_vectors/"+str(main.nodeCount)+"max_diff_vector"+str(case)+".npy")
    axes = plt.gca()    
    axes.set_ylim([0,80])
    n, bins, patches = current_fig.hist(max_diff_vector, 50, density=True, range = (0, 0.1), facecolor='g', alpha=0.75)
plt.savefig('graphs/16green.png', dpi='figure')
            
fig = plt.figure(1, figsize = (20, 20)) 
fig.tight_layout()
for case in range(16):
    disease_count = pow(2, int(case/4))
    high_rest = 1 - 0.2 * int(case%4)
    low_rest = high_rest - 0.2 
    current_fig = fig.add_subplot(4, 4, case+1)
    if case < 4:
        current_fig.set_title("High: "+str('%.1f' % high_rest)+" | Low: "+str('%.1f' % low_rest), fontsize = 20)
    current_fig.set_title(current_fig.get_title()+"\nCase: "+str(case), fontsize = 20)
    if case % 4 == 0:
        current_fig.set_ylabel("Diseased Count: "+str(disease_count) )
    maxScaledDifferences = np.load("graph_vectors/"+str(main.nodeCount)+"maxScaledDifferences"+str(case)+".npy")
    axes = plt.gca()    
    axes.set_ylim([0,10])
    n, bins, patches = plt.hist(maxScaledDifferences, 50, density=True, range = (0, 0.5), facecolor='r', alpha=0.75)
plt.savefig('graphs/16red.png', dpi='figure')
                   
fig = plt.figure(2, figsize = (20, 20)) 
fig.tight_layout()
for case in range(16):
    disease_count = pow(2, int(case/4))
    high_rest = 1 - 0.2 * int(case%4)
    low_rest = high_rest - 0.2 
    current_fig = fig.add_subplot(4, 4, case+1)
    if case < 4:
        current_fig.set_title("High: "+str('%.1f' % high_rest)+" | Low: "+str('%.1f' % low_rest), fontsize = 20)
    current_fig.set_title(current_fig.get_title()+"\nCase: "+str(case), fontsize = 20)
    if case % 4 == 0:
        current_fig.set_ylabel("Diseased Count: "+str(disease_count) )
    train_losses = np.load('datasets/losses/lung'+str(dataSetClass.nodeCount)+'_continuous'+str(case)+'_train_losses.npy')
    test_losses = np.load('datasets/losses/lung'+str(dataSetClass.nodeCount)+'_continuous'+str(case)+'_test_losses.npy')
    current_fig.plot(train_losses)
    current_fig.plot(test_losses)
plt.savefig('graphs/train_test_loss.png', dpi='figure')
             
    