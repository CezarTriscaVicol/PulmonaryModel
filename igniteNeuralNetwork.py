import main
import dataSetClass
from neuralNetworkClass import Net
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import Checkpoint, global_step_from_engine
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
import math
import time

featureCount = dataSetClass.featureCount
outputCount = dataSetClass.outputCount

saveResults = False

def shortToLong(restrictions):
    if main.nodeCount == 128:
        longRestrictions = np.full(main.nodeCount-1, 1.0)
        for i in range(outputCount):
            longRestrictions[i] = restrictions[i]
        for i in range(outputCount+1, main.nodeCount):
            longRestrictions[i-1] = longRestrictions[math.floor(i/2)-1]
        return longRestrictions
    elif main.nodeCount == 1024:
        longRestrictions = np.full(main.nodeCount-1, 1.0)
        for i in range(outputCount):
            longRestrictions[i+14] = restrictions[i]
        for i in range(outputCount+1, main.nodeCount-15):
            longRestrictions[i+14-1] = longRestrictions[math.floor((i+14)/2)-1]
        return longRestrictions

for case in [10]:
    fileName = 'datasets/lung'+str(main.nodeCount)+'_continuous'+str(case)+'.npy'

    dataset = dataSetClass.MyDataset(fileName)

    print(dataset.__len__())

    validation_split = 0.2
    shuffle_dataset = True
    random_seed = 42

    dataset_size = dataset.__len__()
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    epochs = 120
    train_batch_size = 100          #1000
    val_batch_size = train_batch_size
    log_interval = 10
    learning_rate = 0.001           #0.004 
    save_checkpoint = False
    load_checkpoint = False

    means = np.load('datasets/mean_variance/lung'+str(dataSetClass.nodeCount)+'_continuous'+str(case)+'_mean.npy')
    stds = np.load('datasets/mean_variance/lung'+str(dataSetClass.nodeCount)+'_continuous'+str(case)+'_variance.npy')
    means = torch.from_numpy(means)
    stds = torch.from_numpy(stds)

    device = 'cuda'
    means = means.to(device)
    stds = stds.to(device)

    model = Net(means = means, stds = stds)
    model.to(device)

    train_loader = DataLoader(dataset, batch_size = train_batch_size, sampler = train_sampler)
    val_loader = DataLoader(dataset, batch_size = val_batch_size, sampler = valid_sampler)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    test_losses = []
    train_losses = []
    iteration_train_losses = []

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

    val_metrics = {
        "mse": Loss(criterion)
    }
    evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training(trainer):
        batch_loss = trainer.state.output
        e = trainer.state.epoch
        n = trainer.state.max_epochs
        i = trainer.state.iteration
        print(f"Epoch {e}/{n} : {i} - batch loss: {batch_loss}")
        iteration_train_losses.append(trainer.state.output)

    #@trainer.on(Events.EPOCH_COMPLETED)
    #def log_training_results(trainer):
     #   evaluator.run(train_loader)
     #   metrics = evaluator.state.metrics
     #   train_losses.append(metrics['mse'])
     #   print(f"Training Results - Epoch: {trainer.state.epoch} Avg loss: {metrics['mse']:.10f}")

    #@trainer.on(Events.EPOCH_COMPLETED)
    #def log_validation_results(trainer):
    #    evaluator.run(val_loader)
    #    metrics = evaluator.state.metrics
    #    test_losses.append(metrics['mse'])
    #    print(f"Validation Results - Epoch: {trainer.state.epoch} Avg loss: {metrics['mse']:.10f}")


    to_save = {'model': model, 'optimizer': optimizer, 'trainer': trainer}
    checkpoint_dir = "checkpoints/"+str(main.nodeCount)+"_continuous"+str(case)+"/"

    if save_checkpoint:
        checkpoint = Checkpoint(
            to_save,
            checkpoint_dir,
            n_saved=1,
            global_step_transform=global_step_from_engine(trainer),
        )
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint)

    if load_checkpoint:
        checkpoint_fp = checkpoint_dir + "checkpoint_20.pt"
        checkpoint = torch.load(checkpoint_fp, map_location=device)
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    start_time = time.time()

    trainer.run(train_loader, max_epochs=epochs)

    end_time = time.time()
    print(' Execution time = %.6f seconds' % (end_time-start_time))

    plt.figure(1)
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    np.save('datasets/losses/lung'+str(dataSetClass.nodeCount)+'_continuous'+str(case)+'_train_losses.npy', train_losses)
    np.save('datasets/losses/lung'+str(dataSetClass.nodeCount)+'_continuous'+str(case)+'_test_losses.npy', test_losses)

    plt.figure(2)
    plt.plot(iteration_train_losses, label='Training loss')
    #plt.ylim(0, 0.1)
    sample_results = True
    sample_count = 700
    max_sample_count = 10000
    reverse_results = True

    if sample_results:
 
        new_dataset = np.memmap(fileName, dtype = 'float64', mode='r')
        new_dataset.shape = (-1, dataSetClass.columns)
        new_dataset = new_dataset[:max_sample_count]
        new_dataset = np.float32(new_dataset)
        new_dataset = torch.from_numpy(new_dataset)

        X_test = new_dataset[:max_sample_count, :featureCount]
        y_test = new_dataset[:max_sample_count, featureCount:featureCount+outputCount]
        X_test = X_test.to(device)
        y_test_hat = model.forward(X_test)
        X_test = X_test.cpu()
        y_test_hat = y_test_hat.cpu()
        minimum_diff = np.ndarray.max(np.max(np.abs(np.subtract(y_test.detach().numpy()[:1000],y_test_hat.detach().numpy()[:1000])), axis = 1))
        figureNumber = 5
        for i in range(1000):
            if figureNumber >= 9:
                break
            if np.max(np.abs(np.subtract(y_test.detach().numpy()[i],y_test_hat.detach().numpy()[i]))) == minimum_diff:
                plt.figure(figureNumber)
                figureNumber += 1
                plt.plot(y_test_hat.detach().numpy()[i], color = 'red', label='Recomputed')
                plt.plot(y_test.detach().numpy()[i], color = 'green', label='True')
        max_diff_vector = np.ndarray.max(np.subtract(y_test.detach().numpy(),y_test_hat.detach().numpy()), axis = 1)
        if saveResults:
            np.save("graph_vectors/"+str(main.nodeCount)+"max_diff_vector"+str(case), max_diff_vector)
        plt.figure(3)
        n, bins, patches = plt.hist(max_diff_vector, 50, density=True, range = (0, 0.075), facecolor='g', alpha=0.75)
        
        plt.show()
        print("Average of MAX DIFF:", np.mean(max_diff_vector))
        print("Maximum of MAX DIFF:", np.max(max_diff_vector))
        if reverse_results:
            auxx = np.mean(np.abs(np.subtract(y_test_hat[:sample_count].detach().numpy(), y_test[:sample_count].detach().numpy())), axis = 1)
            print("mean loss:", np.mean(auxx))
            y_test_hat = y_test_hat.detach().numpy()
            y_test_hat = [shortToLong(item) for item in y_test_hat]
            recomputedMaxValuesList = main.multiThreadGenericLung(auxG=main.G.copy(), restrictions=y_test_hat[:sample_count], threadCount=7)
            MaxValuesList = X_test.detach().numpy()[:sample_count]
            MaxValueaMeans = np.mean(MaxValuesList, axis = 0)
            MaxValuesVariances = np.sqrt(np.var(MaxValuesList, axis = 0))
            scaledRecomputedMaxValuesList = np.divide(np.subtract(recomputedMaxValuesList, MaxValueaMeans), MaxValuesVariances)
            scaledMaxValuesList = np.divide(np.subtract(MaxValuesList, MaxValueaMeans), MaxValuesVariances)
            differences = np.abs(np.subtract(scaledRecomputedMaxValuesList, scaledMaxValuesList))
            maxScaledDifferences = np.ndarray.max(differences, axis = 1)
            print(np.mean(differences))
            print(np.max(differences))
            print(np.mean(maxScaledDifferences))
            if saveResults:
                np.save("graph_vectors/"+str(main.nodeCount)+"maxScaledDifferences"+str(case), maxScaledDifferences)
            n, bins, patches = plt.hist(maxScaledDifferences, 50, density=True, range = (0, 1), facecolor='r', alpha=0.75)
            plt.show()
            cnt = 10
            for i in range (0, sample_count):
                if maxScaledDifferences[i] > 0.4 and cnt > 0:
                    cnt = cnt - 1
                    plt.figure(i+10)
                    plt.plot(scaledRecomputedMaxValuesList[i], color = 'red', label='Recomputed')
                    plt.plot(scaledMaxValuesList[i], color = 'green', label='True')

