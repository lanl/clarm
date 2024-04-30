''' About this code,
1) A pytorch code
2) cVAE is trained
3) Temporal correlations in the latent space with LSTM 
4) the conditional input is for the encoder only
5) All 15 projections are used
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import os
import sys
sys.path.insert(1, os.path.abspath("E:\MSR\codes\RFLA_VAE_LSTM\pytorch\2dVAE_LSTM_pyt"))

# APIs and inbuilt functions
import numpy as np
import matplotlib.pyplot as plt
from lib.modifyimsize import MODIFYIMSIZE

# information on cuda
device = "cuda" if torch.cuda.is_available() else "cpu"
print('The available device is -->', device)
print("Number of CUDA devices",torch.cuda.device_count())
torch.cuda.set_device(0)
print("Name of CUDA devices",torch.cuda.get_device_name(0))

# calculate the reconstruction error
def reconstruction_error(samples, pred_samples):
    errors = []
    for (image, recon) in zip(samples, pred_samples):
        mse = np.mean((image - recon)**2)
        errors.append(mse)
    return errors

# data normalization
def MinMaxNormalizeData(data):
    # simplified to see the memory utilization at each step
    # dividing an array with a number takes more time and multiple by (1/number)
    minval = np.min(data)
    maxval = np.max(data)
    diff = maxval-minval
    #print('diff is done')
    diffinv = 1/diff
    #print('1/diff is done')
    data_nm_numer = data - minval
    #print('data_nm_numer is done')
    data_nm = data_nm_numer*diffinv
    return data_nm
    
# path for the dataset
im_path = "E://MSR//data//RFLASimulationData//Alan_HPSim_11_13_23//images//"
labels_path = "E://MSR//data//RFLASimulationData//Alan_HPSim_11_13_23//labels//"
im_dir_list = os.listdir(im_path)
labels_dir_list = os.listdir(labels_path)

print("Files and directories in '", im_path, "' :")
print(im_dir_list)

print("Files and directories in '", labels_path, "' :")
print(labels_dir_list)

# loop to import dataset
imgs_all = []
labels_all = []
n = 0
for i in range(len(im_dir_list)):
    n = n+1
    load_imgs = np.load(im_path + im_dir_list[i]).astype('float32')
    load_labels =  np.load(labels_path + labels_dir_list[i]).astype('float32')
    imgs_all.append(load_imgs)
    labels_all.append(load_labels)
    del load_imgs, load_labels

# moveaxis: (p,mod,proj,L,W) -> (p,mod,L,W,proj)
imgs_all_tr = np.concatenate((imgs_all[0],imgs_all[1]),axis=0)
imgs_all_tr = np.moveaxis(imgs_all_tr,2,4)
imgs_all_te = np.moveaxis(imgs_all[2],2,4)

print('Size of the imported train dataset', imgs_all_tr.shape)
print('Size of the imported test dataset',imgs_all_te.shape)

#%% plot samples from dataset - across different runs/perturbations
projection = 12
plot_samples = 48*10
plt.figure(figsize=(15,45))

for i in range(plot_samples):
    cols = 6
    plt.subplot(int(plot_samples/cols) + 1, cols, i + 1)
    plt.imshow(imgs_all_tr[i,0,:,:,projection], aspect='auto', origin='lower', cmap='hsv')
    plt.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
plt.tight_layout()

#%% plot samples from dataset - across different modules
n_mod = 0
run = 0
plot_samples = 15
plt.figure(figsize=(15,15))
for p in range(plot_samples):
    cols = 5
    plt.subplot(int(plot_samples/cols) + 1, cols, p + 1)
    plt.imshow(imgs_all_tr[run,n_mod,:,:,p], aspect='auto', origin='lower', cmap='hsv')
    plt.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
plt.tight_layout()

#%% module numbers as conditional inputs
mod_num = np.arange(0,48,dtype='int64')

mod_num_tr = np.tile(mod_num, (imgs_all_tr.shape[0],1))
mod_num_te = np.tile(mod_num, (imgs_all_te.shape[0],1))

print('module number input (conditional) shape for train', mod_num_tr.shape)
print('module number input (conditional) shape for test',mod_num_te.shape)

print('module number input (first 48)', mod_num_tr[0:48])

#%% Process the dataset (Reshape & Normalize)
# reshape the dimensions of the dataset from 5D to 4D
imgs_all_tr_rs = imgs_all_tr.reshape(imgs_all_tr.shape[0]*imgs_all_tr.shape[1],
                                      imgs_all_tr.shape[2],
                                      imgs_all_tr.shape[3],
                                      imgs_all_tr.shape[4])

imgs_all_te_rs = imgs_all_te.reshape(imgs_all_te.shape[0]*imgs_all_te.shape[1],
                                       imgs_all_te.shape[2],
                                       imgs_all_te.shape[3],
                                       imgs_all_te.shape[4])

mod_num_tr_rs = mod_num_tr.reshape(mod_num_tr.shape[0]* mod_num_tr.shape[1],)
mod_num_te_rs = mod_num_te.reshape(mod_num_te.shape[0]* mod_num_te.shape[1],)

print('Size of resize dataset (trainX)', imgs_all_tr_rs.shape)
print('Size of resize dataset (testX)',imgs_all_te_rs.shape)
print('Size of resize dataset (trainY)', mod_num_tr_rs.shape)
print('Size of resize dataset (testY)',mod_num_te_rs.shape)

del imgs_all_tr, imgs_all_te

#%% modify the image size
# size of the image we want
imsize = 256

if (imsize != 256):
    imgs_all_tr_rs = MODIFYIMSIZE(imgs_all_tr_rs, imsize)
    imgs_all_te_rs = MODIFYIMSIZE(imgs_all_te_rs, imsize)
   
#%% Normalize the data
imgs_all_tr_rs_nm = MinMaxNormalizeData(imgs_all_tr_rs)
imgs_all_te_rs_nm = MinMaxNormalizeData(imgs_all_te_rs)

print('Size of normalized dataset (train)', imgs_all_tr_rs_nm.shape)
print('Size of normalized dataset (test)',imgs_all_te_rs_nm.shape)

del imgs_all_tr_rs, imgs_all_te_rs

#%% plot samples from normalized dataset - different modules == E-z
projection = 11
plot_samples = 48
plt.figure(figsize=(14,5))

for i in range(plot_samples):   
    cols = 12
    plt.subplot(int(plot_samples/cols) + 1, cols, i + 1)
    plt.suptitle('Normalized $E$ - $\phi$ projection',fontsize=16)
    plt.imshow(imgs_all_tr_rs_nm[i,:,:,projection], aspect='auto', origin='lower', cmap='hsv')
    plt.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
    plt.title('Mod.-'+str(i+1),fontsize=10)
    
plt.tight_layout()

#%% Transform the np dataset into tensor
imgs_all_tr_rs_nm = np.moveaxis(imgs_all_tr_rs_nm,3,1)
imgs_all_te_rs_nm = np.moveaxis(imgs_all_te_rs_nm,3,1)

imgs_all_tr_rs_tensor = torch.from_numpy(imgs_all_tr_rs_nm)
imgs_all_te_rs_tensor = torch.from_numpy(imgs_all_te_rs_nm)

mod_num_tr_tensor = torch.from_numpy(mod_num_tr_rs)
mod_num_te_tensor = torch.from_numpy(mod_num_te_rs)

print('Size of final train dataset', imgs_all_tr_rs_tensor.shape)
print('Size of final test dataset',imgs_all_te_rs_tensor.shape)
print('Conditional shape for train', mod_num_tr_tensor.shape)
print('Conditional shape for test',mod_num_te_tensor.shape)

#del imgs_all_tr_rs_nm, imgs_all_te_rs_nm

#%% Tensor dataset &  and use dataloader transforms into a tensor
BATCH_SIZE = 32

from torch.utils.data import Dataset
class DatasetforDataloader(Dataset):
    def __init__(self,X,y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,i):
        # create a tuple
        return self.X[i], self.y[i]
    
# Dataset of dataloader
train_dataset = DatasetforDataloader(imgs_all_tr_rs_tensor,mod_num_tr_tensor)
test_dataset = DatasetforDataloader(imgs_all_te_rs_tensor,mod_num_te_tensor)

# train-test split
train_size = int(0.85 * len(imgs_all_tr_rs_tensor))
val_size = len(imgs_all_tr_rs_tensor) - train_size
trainX, valX  = torch.utils.data.random_split(train_dataset,[train_size, val_size])

## Dataloader wraps an iterable around the Dataset for easy access to the samples.
dataloader_train = DataLoader(trainX, batch_size=BATCH_SIZE, shuffle=True)
dataloader_val = DataLoader(valX, batch_size=BATCH_SIZE, shuffle=False)
dataloader_test = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
dataloader_trainval = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

print('Length of the training set',len(trainX))
print('Length of the validation set',len(valX))

#%% Load untrained ResNet50 model
resnet50_model = models.resnet50()

# Modify the first layer to accept 15 input channels
# The original first layer has 3 input channels (RGB), so we need to change it to accept 15 channels
resnet50_model.conv1 = nn.Conv2d(15, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# Modify the final fully connected layer to match your number of classes
num_classes = 48
resnet50_model.fc = nn.Linear(resnet50_model.fc.in_features, num_classes)

# Move the model to the device
resnet50_model = resnet50_model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet50_model.parameters(), lr=0.001)

#%% Train the model
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassConfusionMatrix

num_epochs = 200

def train_epoch_cvae(model,device,dataloader_train,optimizer,criterion):
        model.train() # Set train mode for model
        train_loss = []
        train_acc = []
        for step, batch in enumerate(dataloader_train):
            x_tr, y_tr = batch
            x_tr, y_tr = x_tr.to(device), y_tr.to(device)
            
            optimizer.zero_grad()
            y_predict = model(x_tr)
            
            loss = criterion(y_predict, y_tr)
            loss.backward()
            optimizer.step()

            mca = MulticlassAccuracy(num_classes=48).to(device)
            acc = mca(y_predict, y_tr)
            
            train_loss.append(loss.item())
            train_acc.append(acc.detach().cpu().numpy())
        return (np.mean(train_loss), np.mean(train_acc))

def test_epoch_cvae(model, device, dataloader_test,criterion):
    model.eval() # Set the eval mode for model
    test_loss = []
    test_acc = []
    with torch.no_grad(): # No need to track the gradients
        for step, batch in enumerate(dataloader_test):
            x_te, y_te = batch
            x_te, y_te = x_te.to(device), y_te.to(device)
            y_predict = model(x_te)
            
            loss = criterion(y_predict, y_te)
            
            mca = MulticlassAccuracy(num_classes=48).to(device)
            acc = mca(y_predict, y_te)
            
            test_loss.append(loss.item())
            test_acc.append(acc.detach().cpu().numpy())
    return (np.mean(test_loss), np.mean(test_acc))
    
diz_loss = {'train_loss':[], 'train_loss_acc':[],
            'val_loss':[],'val_loss_acc':[]}

for epoch in range(num_epochs):
   train_loss, train_acc = train_epoch_cvae(resnet50_model,device,dataloader_train,optimizer,criterion)
   val_loss, val_acc = test_epoch_cvae(resnet50_model,device,dataloader_val,criterion)
   print('\n EPOCH {}/{} \t train loss {} \t val loss {} \t train acc {} \t val acc {}'
         .format(epoch + 1, num_epochs, train_loss, val_loss, train_acc, val_acc))
   diz_loss['train_loss'].append(train_loss)
   diz_loss['train_loss_acc'].append(train_acc)

   diz_loss['val_loss'].append(val_loss)
   diz_loss['val_loss_acc'].append(val_acc)


# Save the trained model
torch.save(resnet50_model.state_dict(), 'resnet50_48classes.pth')

#%% plot the loss and accuracy
#loss
plt.figure(figsize=(10,8))
plt.plot(diz_loss['train_loss'], '-ok', label='Train')
plt.plot(diz_loss['val_loss'], '-^r', label='Valid')
plt.xlabel('Epoch',fontsize=20)
plt.ylabel('Average Loss',fontsize=20)
plt.legend(["tr_loss", "val_loss"])
plt.title('Training & Validation loss', fontsize = 20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(0, 1e-4)
plt.show()

# acc
plt.figure(figsize=(10,8))
plt.plot(diz_loss['train_loss_acc'], '-ok', label='Train',)
plt.plot(diz_loss['val_loss_acc'], '-^r', label='Valid')
plt.xlabel('Epoch',fontsize=20)
plt.ylabel('Average Acc',fontsize=20)
plt.legend(["tr_acc", "val_acc"])
plt.title('Training & Validation loss', fontsize = 20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(0.9,1.01)
plt.show()

#%% predictions on test set
test_loss = []
test_acc = []
test_precision = []
test_recall = []
with torch.no_grad(): # No need to track the gradients
    for step, batch in enumerate(dataloader_test):
        x_te, y_te = batch
        x_te, y_te = x_te.to(device), y_te.to(device)
        y_predict = resnet50_model(x_te)
        _, y_predict_class = torch.max(y_predict, 1)
        
        loss = criterion(y_predict, y_te)
        
        mca = MulticlassAccuracy(num_classes=48).to(device)
        acc = mca(y_predict, y_te)
        
        pre = MulticlassPrecision(num_classes=48).to(device)
        precision = pre(y_predict, y_te)
        
        rec = MulticlassRecall(num_classes=48).to(device)
        recall = rec(y_predict, y_te)
        
        test_loss.append(loss.item())
        test_acc.append(acc.detach().cpu().numpy())
        test_precision.append(precision.detach().cpu().numpy())
        test_recall.append(recall.detach().cpu().numpy())
        
test_loss, test_acc = np.array(test_loss), np.array(test_acc)
test_precision, test_recall = np.array(test_precision), np.array(test_recall)

print('Accuracy for the test set', test_acc)

#%% test the classifier with test images
imgs_all_te_nm = imgs_all_te_rs_nm.reshape(100,48,15,256,256)
n_mod = 40
run = np.random.randint(0,100,1).item()
plot_samples = 15
plt.figure(figsize=(15,15))
dataplot = imgs_all_te_nm[run,n_mod,:,:,:]
for p in range(plot_samples):
    cols = 5
    plt.subplot(int(plot_samples/cols) + 1, cols, p + 1)
    plt.imshow(dataplot[p], aspect='auto', origin='lower', cmap='hsv')
    plt.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
plt.tight_layout()

dataplot = torch.from_numpy(dataplot[np.newaxis,:,:,:])
y_pred_te = resnet50_model(dataplot.to(device))
_, y_pred_te_class = torch.max(y_pred_te, 1)

print('Run number = ', run)
print('The true class = ', n_mod)
print('The predicted class = ', y_pred_te_class.item())