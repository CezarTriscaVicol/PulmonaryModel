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

featureCount = dataSetClass.featureCount
outputCount = dataSetClass.outputCount

case = 2
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

epochs = 2
train_batch_size = 10000
val_batch_size = 10000
log_interval = int(dataset.__len__() / (train_batch_size*5))
save_checkpoint = True
load_checkpoint = False

model = Net()
device = 'cuda'
model.to(device)

train_loader = DataLoader(dataset, batch_size = train_batch_size, sampler = train_sampler)
val_loader = DataLoader(dataset, batch_size = val_batch_size, sampler = valid_sampler)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
test_losses = []
train_losses = []
iteration_train_losses = []

trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

val_metrics = {
    "mse": Loss(criterion)
}
evaluator = create_supervised_evaluator(
    model, metrics=val_metrics, device=device)

@trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
def log_training(trainer):
    batch_loss = trainer.state.output
    e = trainer.state.epoch
    n = trainer.state.max_epochs
    i = trainer.state.iteration
    print(f"Epoch {e}/{n} : {i} - batch loss: {batch_loss}")
    iteration_train_losses.append(trainer.state.output)

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    train_losses.append(metrics['mse'])
    print(f"Training Results - Epoch: {trainer.state.epoch} Avg loss: {metrics['mse']:.10f}")

@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    test_losses.append(metrics['mse'])
    print(f"Validation Results - Epoch: {trainer.state.epoch} Avg loss: {metrics['mse']:.10f}")


to_save = {'model': model, 'optimizer': optimizer, 'trainer': trainer}
checkpoint_dir = "checkpoints/"+str(main.nodeCount)+"_continuous"+str(case)+"/"

if save_checkpoint:
    checkpoint = Checkpoint(
        to_save,
        checkpoint_dir,
        n_saved=1,
        global_step_transform=global_step_from_engine(trainer),
    )
    evaluator.add_event_handler(Events.COMPLETED, checkpoint)

if load_checkpoint:
    checkpoint_fp = checkpoint_dir + "checkpoint_200.pt"
    checkpoint = torch.load(checkpoint_fp, map_location=device)
    Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

for stepp in range(100):
    trainer.run(train_loader, max_epochs=epochs)

plt.figure(1)
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')

plt.figure(2)
plt.plot(iteration_train_losses, label='Training loss')

sample_results = True 
sample_count = 7*50

auxx = np.memmap(fileName, mode='r', dtype = 'float64')
auxx.resize((int(auxx.shape[0]/dataSetClass.columns), dataSetClass.columns))
auxx = np.asarray(auxx[:sample_count].tolist())
X_test = auxx[:,:featureCount]
y_test = auxx[:,featureCount:]
if(dataSetClass.swap):
    X_test = auxx[:,outputCount:]
    y_test = auxx[:,:outputCount]
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

model.to('cpu')
y_test_hat = model.forward(X_test[:sample_count])
for i in range(0, 2):
    print("TEST:",y_test_hat[i].detach().numpy())
    print("TRUE:",y_test[i].detach().numpy())
    print("DIFF:",np.subtract(y_test[i].detach().numpy(),y_test_hat[i].detach().numpy()))

if(dataSetClass.swap):
    scaled_mean = 0
    scaled_variance = 0
    scaled_max = 0
    for i in range(0, 1000):
        scaled_mean += np.mean(np.divide(y_test.detach().numpy(), np.abs(np.subtract(y_test.detach().numpy(),y_test_hat.detach().numpy()))))
        scaled_variance += np.divide(np.mean(y_test_hat.detach().numpy()), np.subtract(np.max(y_test.detach().numpy()),np.min(y_test.detach().numpy())))
        scaled_max += np.max(np.divide(y_test.detach().numpy(), np.abs(np.subtract(y_test.detach().numpy(),y_test_hat.detach().numpy()))))
    scaled_mean /= 1000
    scaled_variance /= 1000
    scaled_max /= 1000

    print("Mean:",scaled_mean)
    print("Variance:",scaled_variance)
    print("Max:",scaled_max)

if sample_results:

    sum = 0
    for i in range(0, sample_count):
        auxx = np.mean(np.abs(np.subtract(
            y_test_hat[i].detach().numpy(), y_test[i].detach().numpy())))
        sum += auxx
    print("mean loss:", sum/sample_count)
    y_test_hat = y_test_hat.detach().numpy()
    recomputedMaxValuesList = main.multiThreadGenericLung(auxG=main.G.copy(), restrictions=y_test_hat, threadCount=7)
    scaled_mean_ = 0
    scaled_variance_ = 0
    scaled_max_ = 0
    absolute_mean_ = 0
    absolute_variance_ = 0
    absolute_max_ = 0
    absolute_max_val = 0
    absolute_min_val = 1
    for j in range(sample_count):
        currentTest = X_test[j].detach().numpy()
        recomputedMaxValues = recomputedMaxValuesList[j]
        scaled_mean_ += np.mean(
            np.divide(np.abs(np.subtract(recomputedMaxValues, currentTest)), currentTest))
        scaled_variance_ += (np.max(currentTest) -
                            np.min(currentTest))/np.mean(currentTest)
        scaled_max_ = max(scaled_max_, np.max(
            np.divide(np.abs(np.subtract(recomputedMaxValues, currentTest)), currentTest)))
        absolute_mean_ += np.mean(np.abs(np.subtract(recomputedMaxValues, currentTest)))
        absolute_variance_ += np.max(currentTest)-np.min(currentTest)
        absolute_max_ = max(absolute_max_, np.max(
            np.abs(np.subtract(recomputedMaxValues, currentTest))))
        absolute_max_val = max(absolute_max_val, np.max(currentTest))
        absolute_min_val = min(absolute_min_val, np.min(currentTest))

    scaled_mean_ /= sample_count
    scaled_variance_ /= sample_count
    absolute_mean_ /= sample_count
    absolute_variance_ /= sample_count

    print("S Average natural variance in volume, scaled:", scaled_variance_)
    print("S Mean difference in volume, scaled:", scaled_mean_)
    print("S Maximum difference in volume, scaled:", scaled_max_)
    print("S Ratio between variance and maximum:", scaled_variance_/scaled_max_)
    print("S Ratio between variance and mean:", scaled_variance_/scaled_mean_)
    print("")
    print("M Maximum absolute volume:", absolute_max_val)
    print("M Minimum absolute volume:", absolute_min_val)
    print("")
    print("A Average natural variance in volume, absolute:", absolute_variance_)
    print("A Mean difference in volume, absolute:", absolute_mean_)
    print("A Maximum difference in volume, absolute:", absolute_max_)
    print("A Ratio between variance and maximum, absolute:",
        absolute_variance_/absolute_max_)
    print("A Ratio between variance and mean, absolute:",
        absolute_variance_/absolute_mean_)
