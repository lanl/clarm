''' About this code,
1) A pytorch code
2) CVAE is trained where module number acts as conditional to encoder
3) Temporal correlations are learned in the latent space with LSTM 
5) Autoregressive loop help forecast projections in all modules
'''

import torch
import torch.optim as optim
import os
import sys
sys.path.insert(1, os.path.abspath("E:\MSR\codes\RFLA_VAE_LSTM_pyt"))

# APIs and inbuilt functions
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from lib.modifyimsize import MODIFYIMSIZE

# information on cuda
print("Number of CUDA devices",torch.cuda.device_count())
print("Name of CUDA device - 1 ==>",torch.cuda.get_device_name(0))
print("Name of CUDA device - 2 ==>",torch.cuda.get_device_name(1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device1 = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
device2 = torch.device("cuda:1")
    
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

# concatentate the appended dataset
imgs_all_tr = np.concatenate((imgs_all[0],imgs_all[1]),axis=0)

# moveaxis: (p,mod,proj,L,W) -> (p,mod,L,W,proj)
# modify images and normalize require moveaxis
imgs_all_tr = np.moveaxis(imgs_all_tr,2,4)
imgs_all_te = np.moveaxis(imgs_all[2],2,4)

print('Size of the imported train dataset', imgs_all_tr.shape)
print('Size of the imported test dataset',imgs_all_te.shape)

#%% =====>>>> DATASET VISUALIZATION <<<<======
############################################################################################
#########################  DATASET VISUALIZATION ###########################################
############################################################################################

#%% plot samples from dataset - across different runs/perturbations (axis=0)
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
#plt.savefig('dataset_runs_proj11.png', dpi=300)

#%% plot samples from dataset - across different modules
n_mod = 1
run = 0
plot_samples = 15
plt.figure(figsize=(15,15))
for p in range(plot_samples):
    cols = 5
    plt.subplot(int(plot_samples/cols) + 1, cols, p + 1)
    plt.imshow(np.log(1+imgs_all_tr[run,n_mod,:,:,p]), aspect='auto', origin='lower', cmap='plasma')
    plt.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
plt.tight_layout()
#plt.savefig('dataset_runs_allproj_mod.png', dpi=300)

#%% plot samples from dataset - different modules - different projections (E-phi)
E_ref = [5.35508654, 41.30085241, 72.59726239, 99.55659545, 113.30694285,
        126.54444021, 139.06689037, 153.99711291, 167.78542056, 182.78942052,
        196.36477222, 212.36085887, 225.5727351, 242.05917683, 257.87629262,
        270.56978027, 287.61654992, 304.05749827, 317.68888699, 334.16846795,
        351.99214741, 366.78784088, 381.9247197, 398.52202592, 415.99627685,
        430.9371102, 446.40440063, 464.28822739, 482.19924841, 497.70466852,
        512.78249953, 529.80507787, 548.09348445, 566.19867131, 582.82614372,
        597.95315468, 614.21267512, 632.03148033, 650.15885225, 667.67175433,
        683.26887754, 698.57431461, 715.5213991, 733.9661626, 751.98340543,
        769.06200623, 784.84248942, 800.09776699]

phi_ref = [11280.83108553, 35524.52623656, 49838.59318148, 61305.903995,
           288361.38037085, 319788.38387692, 351209.13837744, 381470.37516268,
           411729.70835502, 440524.90018308, 469362.37242238, 498138.63636995,
           523347.67225599, 547484.83300091, 571555.07608093, 595678.48363986,
           619799.63361058, 643884.756519, 667272.15623005, 690674.49256342,
           714044.80546704, 737047.56147925, 760089.84080003, 782414.85757303,
           804711.56299811, 827019.32420482, 849350.09611553, 871678.29433691,
           893998.67106446, 915579.15614886, 937194.61074431, 958443.63612275,
           979675.77897511, 1000909.73863817, 1022126.5531958, 1042646.95242532,
           1063171.49187095, 1083695.10086393, 1104198.70898035, 1124705.97951749,
           1145215.59676875, 1165731.03826898, 1186267.48596319, 1206785.63240674,
           1226584.02594906, 1246374.38559172, 1266166.68815084, 1284803.08186162]

scale1 = 0.01
scale2 = 0.0001

E_ref2 = np.array(E_ref)*scale1
E_ref2 = np.round(E_ref2,1)
phi_ref2 = np.array(phi_ref)*scale2
phi_ref2 = np.round(phi_ref2,1)

phi_b = 60*scale1
E_b = 1.3*0.01*E_ref2*scale2

phi_lb, phi_ub = phi_ref2-60, phi_ref2+60
phi_lb, phi_ub = np.round(phi_lb,1),np.round(phi_ub,1)

E_lb, E_ub = E_ref2-E_b, E_ref2+E_b
E_lb, E_ub = np.round(E_lb,1),np.round(E_ub,1)

projection = 11 # E-phi
plot_samples = 48
plt.figure(figsize=(14,8))
dataplot = imgs_all_tr
for i in range(plot_samples):  
    cols = 8
    plt.subplot(int(plot_samples/cols) + 1, cols, i + 1)
    plt.imshow(dataplot[0,i,:,:,projection], aspect='auto', origin='lower', cmap='hsv')
    #plt.tick_params(left = False, right = False , labelleft = False ,
                    #labelbottom = False, bottom = False)
    plt.title('Mod.-'+str(i+1), fontsize = 12)
    if (i == 0 or i == 8 or i == 16 or i == 24 or i == 32 or i == 40):
        plt.ylabel('E (MeV*100)',fontsize=10)
    
    if (i>40 and i<48):
        plt.xlabel('$\phi (rad*10^4)$',fontsize=10)
    
    xticks = [phi_lb[i], phi_ref2[i], phi_ub[i]]
    yticks = [E_lb[i], E_ref2[i], E_ub[i]]
    
    xticklabels = [str(xticks[0]), str(xticks[1]), str(xticks[2])]
    yticklabels = [str(yticks[0]), str(yticks[1]), str(yticks[2])]
    
    plt.xticks([0,128,256], xticklabels, fontsize=12)
    plt.yticks([0,128,256], yticklabels, fontsize=12)
    
plt.tight_layout()

#%% plot samples from dataset - different modules - different projections (E-phi)
projection = 11 # E-phi
plot_samples = 48
plt.figure(figsize=(15,7))
dataplot = imgs_all_tr
for i in range(plot_samples):  
    cols = 12
    plt.subplot(int(plot_samples/cols) + 1, cols, i + 1)
    plt.suptitle('$E$ - $\phi$ projection',fontsize=20)
    plt.imshow(dataplot[0,i,:,:,projection], aspect='auto', origin='lower', cmap='hsv')
    #plt.tick_params(left = False, right = False , labelleft = False ,
                    #labelbottom = False, bottom = False)
    plt.title('Mod.-'+str(i+1), fontsize = 12)
    if (i == 0 or i == 12 or i == 24 or i == 36 or i == 48):
        plt.ylabel('% $E_{r}$ (MeV)',fontsize=12)
        yticks = [-1.3, 0, 1.3]
        yticklabels = [str(yticks[0]), str(yticks[1]), str(yticks[2])]
        plt.yticks([0,128,256], yticklabels, fontsize=10)
    else:
        plt.yticks([],fontsize=10)
    
    if (i>35 and i<48):
        plt.xlabel('$\Delta \phi (deg)$',fontsize=12)
        xticks = [-60, 0, 60]
        xticklabels = [str(xticks[0]), str(xticks[1]), str(xticks[2])]
        plt.xticks([0,128,256], xticklabels, fontsize=10)
    else:
        plt.xticks([],fontsize=10) 
    
    plt.subplots_adjust(wspace=0.23, hspace=0.32)
    
plt.savefig('dataset_modules_4x12_proj_'+str(projection)+'.png', bbox_inches='tight',dpi=300)

#%% plot samples from dataset - different modules - 3 projections (E-phi) ==> 2 x 6
path = 'E:/MSR/codes/RFLA_VAE_LSTM_pyt/figures/fig_data/'

projection = 11 # E-phi
plot_samples = 12
pltfig = [0,1,2,3,10,15,20,25,30,35,40,47]
plt.figure(figsize=(15,7))
dataplot = imgs_all_tr
#dataplot = np.log(1+imgs_all_tr)
for i in range(plot_samples):  
    cols = 6
    plt.subplot(int(plot_samples/cols) + 1, cols, i + 1)
    plt.suptitle('$E$ - $\phi$ projection',fontsize=20)
    plt.imshow(dataplot[0,pltfig[i],:,:,projection], aspect='auto', origin='lower', cmap='plasma')
    #plt.tick_params(left = False, right = False , labelleft = False ,
                    #labelbottom = False, bottom = False)
    plt.title('Mod.-'+str(pltfig[i]+1), fontsize = 16)
    if (i == 0 or i == 6):
        plt.ylabel('% $E_{r}$ (MeV)',fontsize=16)
        yticks = [-1.3, 0, 1.3]
        yticklabels = [str(yticks[0]), str(yticks[1]), str(yticks[2])]
        plt.yticks([0,128,256], yticklabels, fontsize=16)
    else:
        plt.yticks([],fontsize=10)
    
    if (i>5 and i<13):
        plt.xlabel('$\Delta \phi (deg)$',fontsize=16)
        xticks = [-60, 0, 60]
        xticklabels = [str(xticks[0]), str(xticks[1]), str(xticks[2])]
        plt.xticks([0,128,256], xticklabels, fontsize=16)
    else:
        plt.xticks([],fontsize=10) 
    
    plt.subplots_adjust(wspace=0.23, hspace=0.32)

plt.savefig(path + 'dataset_modules_2x6_proj_'+str(projection) + '_unlog' +'.png', bbox_inches='tight',dpi=300)
   
#%% plot samples from dataset - different modules - 3 projections (x-y) ==> 2 x 6
projection = 1 # x-y
plot_samples = 12
pltfig = [0,1,2,3,10,15,20,25,30,35,40,47]
plt.figure(figsize=(15,7))

for i in range(plot_samples):  
    cols = 6
    plt.subplot(int(plot_samples/cols) + 1, cols, i + 1)
    plt.suptitle('$x - y$ projection',fontsize=20)
    plt.imshow(dataplot[0,pltfig[i],:,:,projection], aspect='auto', origin='lower', cmap='plasma')
    #plt.tick_params(left = False, right = False , labelleft = False ,
                    #labelbottom = False, bottom = False)
    plt.title('Mod.-'+str(pltfig[i]+1), fontsize = 16)
    if (i == 0 or i == 6):
        plt.ylabel('x(cm)',fontsize=16)
        yticks = [-3, 0, 3]
        yticklabels = [str(yticks[0]), str(yticks[1]), str(yticks[2])]
        plt.yticks([0,128,256], yticklabels, fontsize=16)
    else:
        plt.yticks([],fontsize=10)
    
    if (i>5 and i<13):
        plt.xlabel('$y(cm)$',fontsize=16)
        xticks = [-3, 0, 3]
        xticklabels = [str(xticks[0]), str(xticks[1]), str(xticks[2])]
        plt.xticks([0,128,256], xticklabels, fontsize=16)
    else:
        plt.xticks([],fontsize=10)
    
    plt.subplots_adjust(wspace=0.23, hspace=0.32)
        
plt.savefig(path + 'dataset_modules_2x6_proj_'+str(projection) + '_unlog' +'.png',bbox_inches='tight', dpi=300)

#%% plot samples from dataset - different modules - 3 projections (x'-y') ==> 2 x 6
projection = 12 # x'-y'
plot_samples = 12
pltfig = [0,1,2,3,10,15,20,25,30,35,40,47]
plt.figure(figsize=(15,7))
for i in range(plot_samples):  
    cols = 6
    plt.subplot(int(plot_samples/cols) + 1, cols, i + 1)
    plt.suptitle('$x^\prime$ - $y^\prime$ projection',fontsize=20)
    plt.imshow(dataplot[0,pltfig[i],:,:,projection], aspect='auto', origin='lower', cmap='plasma')
    #plt.tick_params(left = False, right = False , labelleft = False ,
                    #labelbottom = False, bottom = False)
    plt.title('Mod.-'+str(pltfig[i]+1), fontsize = 16)
    if (i == 0 or i == 6):
        plt.ylabel('$x^\prime$ (mrad)',fontsize=16)
        yticks = [-10, 0, 10]
        yticklabels = [str(yticks[0]), str(yticks[1]), str(yticks[2])]
        plt.yticks([0,128,256], yticklabels, fontsize=16)
    else:
        plt.yticks([],fontsize=10)
    
    if (i>5 and i<13):
        plt.xlabel('$y^\prime$ (mrad)',fontsize=16)
        xticks = [-10, 0, 10]
        xticklabels = [str(xticks[0]), str(xticks[1]), str(xticks[2])]
        plt.xticks([0,128,256], xticklabels, fontsize=16)
    else:
        plt.xticks([],fontsize=10)
        
    plt.subplots_adjust(wspace=0.23, hspace=0.32)
    
plt.savefig(path + 'dataset_modules_2x6_proj_'+str(projection) + '_unlog' +'.png',bbox_inches='tight', dpi=300)

#%% =====>>>> DATASET PROCESSING <<<<======
############################################################################################
#########################  DATASET PRCOESSING #############################################
############################################################################################

#%% module numbers as conditional inputs
mod_num = np.arange(0,48,dtype='float32')
mod_num_nm = mod_num[:,np.newaxis]/np.max(mod_num)

mod_num_tr = np.tile(mod_num_nm, (imgs_all_tr.shape[0],1))
mod_num_te = np.tile(mod_num_nm, (imgs_all_te.shape[0],1))

print('module number input (conditional) shape for train', mod_num_tr.shape)
print('module number input (conditional) shape for test',mod_num_te.shape)

print('module number input (first 48)', mod_num_tr[0:48])

#%% Process the dataset (Reshape & Normalize)
# size of the image we want
imsize = 256

# reshape the dimensions of the dataset from 5D to 4D
imgs_all_tr_rs = imgs_all_tr.reshape(imgs_all_tr.shape[0]*imgs_all_tr.shape[1],
                                      imgs_all_tr.shape[2],
                                      imgs_all_tr.shape[3],
                                      imgs_all_tr.shape[4])

imgs_all_te_rs = imgs_all_te.reshape(imgs_all_te.shape[0]*imgs_all_te.shape[1],
                                       imgs_all_te.shape[2],
                                       imgs_all_te.shape[3],
                                       imgs_all_te.shape[4])

print('Size of reshaped dataset (train)', imgs_all_tr_rs.shape)
print('Size of reshaped dataset (test)',imgs_all_te_rs.shape)

#del imgs_all_tr, imgs_all_te

#%% modify the image size and normalize the data
if (imsize != 256):
    imgs_all_tr_rs = MODIFYIMSIZE(imgs_all_tr_rs, imsize)
    imgs_all_te_rs = MODIFYIMSIZE(imgs_all_te_rs, imsize)

# function for data normalization
def MinMaxNormalizeData(data):
    # simplified to see the memory utilization at each step
    # dividing an array with a number takes more time and multiple by (1/number)
    minval = np.min(data)
    maxval = np.max(data)
    diff = maxval-minval
    diffinv = 1/diff
    data_nm_numer = data - minval
    data_nm = data_nm_numer*diffinv
    return data_nm, minval, maxval

def UnNormalize(data, minval, maxval):
    diff = maxval-minval
    data_un = data*diff + minval
    return data_un

# Normalize the data
imgs_all_tr_rs_nm, minval_tr, maxval_tr = MinMaxNormalizeData(imgs_all_tr_rs)
imgs_all_te_rs_nm, minval_te, maxval_te = MinMaxNormalizeData(imgs_all_te_rs)

print('Size of resized & normalized dataset (train)', imgs_all_tr_rs_nm.shape)
print('Size of resized & normalized dataset (test)',imgs_all_te_rs_nm.shape)

del imgs_all_tr_rs, imgs_all_te_rs

#%% plot samples from normalized dataset - different modules == E-z
projection = 11
plot_samples = 48
plt.figure(figsize=(14,5))
dataplot = imgs_all_tr_rs_nm

for i in range(plot_samples):   
    cols = 12
    plt.subplot(int(plot_samples/cols) + 1, cols, i + 1)
    plt.suptitle('Normalized $E$ - $\phi$ projection',fontsize=16)
    plt.imshow(dataplot[i,:,:,projection], aspect='auto', origin='lower', cmap='hsv')
    plt.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
    plt.title('Mod.-'+str(i+1),fontsize=10)
    
plt.tight_layout()
del dataplot

#%% Some data analysis = SSID and MSE across different perturbations
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

module = 40
ssim_da_all = []
mse_da_all=[]
for i in range(imgs_all_tr.shape[0]-1):
    img_1 = imgs_all_tr[0,module,:,:,projection]
    img_2 = imgs_all_tr[i+1,module,:,:,projection]
    ssim_da = ssim(img_1, img_2, data_range=img_1.max() - img_2.min())
    mse_da = mean_squared_error(img_1,img_2)
    ssim_da_all.append(ssim_da)
    mse_da_all.append(mse_da)
    
plt.figure(figsize = (30, 6))
plt.plot(ssim_da_all,'--or')
plt.ylabel("SSIM",fontsize=20)
plt.yticks(fontsize=20)

plt.figure(figsize = (30, 6))
plt.plot(mse_da_all,'--or')
plt.ylabel("MSE",fontsize=20)
plt.yticks(fontsize=20)

#%% Some data analysis = SSID and MSE across the modules
projection = 11

ssim_matrix = np.zeros((48,48))
mse_matrix = np.zeros((48,48))

data_matrix = imgs_all_tr_rs_nm[0:48,:,:,projection]

for i in range(48):
    for j in range(48):
        img_1 = data_matrix[i]
        img_2 = data_matrix[j]
        if j >= i:
            maxval = img_2.max()
            minval = img_2.min()
        else:
            maxval = img_1.max()
            minval = img_1.min()
        ssim_da = ssim(img_1, img_2, data_range=maxval - minval)
        mse_da = mean_squared_error(img_1,img_2)
        ssim_matrix[i,j] = ssim_da
        mse_matrix[i,j] = mse_da

plt.figure(figsize=(20, 20))
plt.imshow(ssim_matrix,cmap='jet')
plt.colorbar()
plt.xticks(fontsize=20)
plt.xticks(np.arange(0,48),fontsize=12)
plt.yticks(np.arange(0,48),fontsize=12)

plt.figure(figsize=(20, 20))
plt.imshow(mse_matrix,cmap='jet')
plt.colorbar()
plt.xticks(np.arange(0,48),fontsize=12)
plt.yticks(np.arange(0,48),fontsize=12)

del data_matrix

#%% =====>>>> TRAINING CVAE <<<<======
############################################################################################
#########################  CVAE TRAINING  ##################################################
############################################################################################

#%% Transform the np dataset into tensor
imgs_all_tr_rs_nm = np.moveaxis(imgs_all_tr_rs_nm,3,1)
imgs_all_te_rs_nm = np.moveaxis(imgs_all_te_rs_nm,3,1)

imgs_all_tr_rs_tensor = torch.from_numpy(imgs_all_tr_rs_nm)
imgs_all_te_rs_tensor = torch.from_numpy(imgs_all_te_rs_nm)

mod_num_tr_tensor = torch.from_numpy(mod_num_tr)
mod_num_te_tensor = torch.from_numpy(mod_num_te)

print('Size of final train dataset', imgs_all_tr_rs_tensor.shape)
print('Size of final test dataset',imgs_all_te_rs_tensor.shape)
print('Conditional shape for train', mod_num_tr_tensor.shape)
print('Conditional shape for test',mod_num_te_tensor.shape)

del imgs_all_tr_rs_nm, imgs_all_te_rs_nm
            
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
dataloader_train = DataLoader(trainX, batch_size = BATCH_SIZE, shuffle=True)
dataloader_val = DataLoader(valX, batch_size = BATCH_SIZE, shuffle=False)
dataloader_test = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False)
dataloader_trainval = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=False)

print('Length of the training set',len(trainX))
print('Length of the validation set',len(valX))

#%% CVAE model
# Move the model to GPU
# Initialize the autoencoder
from lib.cvae_pyt import cVAE
from lib.elbo_loss import ELBO

imch = 15
f1,f2,f3,f4,f5 = 32,64,128,256,512
neurons = 256
branchneurons = 32
combinedneurons = neurons + branchneurons
latentdim = 8
strides = 2
nfilters = 5
imfinal1 = int(imsize/(strides**nfilters))
imfinal2 = int(imsize/(strides**nfilters))

cvae15 = cVAE(imch,f1,f2,f3,f4,f5,
              neurons,branchneurons,combinedneurons,
              latentdim,imfinal1,imfinal2,device1)
#cvae15 = torch.nn.DataParallel(cvae15)
cvae15 = cvae15.to(device1)

print("Num params encoder: ", sum(p.numel() for p in cvae15.parameters()))
print('Model architecture', cvae15)

# Define the loss function and optimizer
optimizer = optim.Adam(cvae15.parameters(), lr=0.001)
beta = 1

#%% Training
num_epochs = 1500

def train_epoch_cvae(model,device,dataloader_train,optimizer):
        model.train() # Set train mode for model
        train_loss = []
        train_loss_bce = []
        train_loss_kl = []
        for step, batch in enumerate(dataloader_train):
            x_tr, c_tr = batch
            x_tr = x_tr.to(device)
            c_tr = c_tr.to(device)
            optimizer.zero_grad()
            x_hat, mean, log_var, z = model(x_tr,c_tr)
            loss, bce, kl = ELBO(x_tr, x_hat, mean, log_var, beta)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            train_loss_bce.append(bce.item())
            train_loss_kl.append(kl.item())
        return (np.mean(train_loss),np.mean(train_loss_bce),np.mean(train_loss_kl))

def test_epoch_cvae(model, device, dataloader_test):
    model.eval() # Set the eval mode for model
    test_loss = []
    test_loss_bce = []
    test_loss_kl = []
    with torch.no_grad(): # No need to track the gradients
        for step, batch in enumerate(dataloader_test):
            x_val, c_val = batch
            x_val = x_val.to(device)
            c_val = c_val.to(device)
            x_hat, mean, log_var, z = model(x_val,c_val)
            loss, bce, kl = ELBO(x_val, x_hat, mean, log_var, beta)
            test_loss.append(loss.item())
            test_loss_bce.append(bce.item())  
            test_loss_kl.append(kl.item())  
    return (np.mean(test_loss),np.mean(test_loss_bce),np.mean(test_loss_kl))
    
diz_loss = {'train_loss':[], 'train_loss_bce':[],'train_loss_kl':[],
            'val_loss':[],'val_loss_bce':[],'val_loss_kl':[]}

for epoch in range(num_epochs):
   train_loss, tr_bce, tr_kl = train_epoch_cvae(cvae15,device1,dataloader_train,optimizer)
   val_loss, val_bce, val_kl = test_epoch_cvae(cvae15,device1,dataloader_val)
   print('\n EPOCH {}/{} \t train loss {} \t val loss {} \t train recon {} \t val recon {} \ttrain kl {} \t val kl {}'
         .format(epoch + 1, num_epochs,train_loss,val_loss,tr_bce,val_bce,tr_kl,val_kl))
   diz_loss['train_loss'].append(train_loss)
   diz_loss['train_loss_bce'].append(tr_bce)
   diz_loss['train_loss_kl'].append(tr_kl)
   diz_loss['val_loss'].append(val_loss)
   diz_loss['val_loss_bce'].append(val_bce)
   diz_loss['val_loss_kl'].append(val_kl)

# Save the model
torch.save(cvae15.state_dict(), 'cvae_15ch_model.pth') 

#%% load the model
cvae15.load_state_dict(torch.load('cvae_15ch_model.pth'))
cvae15.eval()

#%% cVAE training and test loss
# Plot losses
plt.figure(figsize=(10,8))
plt.plot(diz_loss['train_loss'], '-ok', label='Train',)
plt.plot(diz_loss['val_loss'], '-^r', label='Valid')
plt.xlabel('Epoch',fontsize=20)
plt.ylabel('Average Loss',fontsize=20)
plt.legend(["tr_total", "val_total"])
plt.title('Training & Validation loss', fontsize = 20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(49000, 60000)
plt.show()

# Plot losses on semilog scale
plt.figure(figsize=(10,8))
plt.plot(diz_loss['train_loss_bce'],'-ob', label='Train bce')
plt.plot(diz_loss['val_loss_bce'], '-^r', label='Valid bce')
plt.xlabel('Epoch',fontsize=20)
plt.ylabel('Reconstruction loss',fontsize=20)
plt.legend(["tr_bce", "val_bce"],fontsize=20)
plt.title('Training & Validation loss', fontsize = 20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(49000, 60000)
plt.show()

# Plot losses on semilog scale
plt.figure(figsize=(10,8))
plt.plot(diz_loss['train_loss_kl'],'-ob', label='Train kl')
plt.plot(diz_loss['val_loss_kl'],'-^r', label='Valid kl')
plt.xlabel('Epoch',fontsize=20)
plt.ylabel('KL divergence loss',fontsize=20)
plt.legend(["tr_kl" , "val_kl"],fontsize=20)
plt.title('Training & Validation loss', fontsize = 20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(200,500)
plt.show()

#%% =====>>>> RECONSTRUCTION RESULTS <<<<======
############################################################################################
#########################  RECONSTRUCTION PART #############################################
############################################################################################

#%% Collect all reconstruction images for training and testing data
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

# calculate the reconstruction error
def reconstruction_error(samples, pred_samples):
    errors = []
    for (image, recon) in zip(samples, pred_samples):
        mse = np.mean((image - recon)**2)
        errors.append(mse)
    return errors

# calculate mse and ssim projection wise
def mse_ssim_projwise(realdata,gendata):
    ssim_app = []
    mse_app = []
    for p in range(realdata.shape[0]):
        real = realdata[p]
        gen = gendata[p]
        sim = ssim(real, gen, data_range=real.max() - real.min())
        mse = mean_squared_error(real, gen)
        ssim_app.append(sim)
        mse_app.append(mse)
    return np.array(mse_app), np.array(ssim_app)

# calculate mse and ssim sample wise
def mse_ssim_samplewise(data,recondata):
    mse2_app, ss2_app = [], []
    for s in range(data.shape[0]):
        org = data[s,:,:,:]
        recon = recondata[s,:,:,:]
        mse, ss = mse_ssim_projwise(org, recon)
        mse2_app.append(mse)
        ss2_app.append(ss)
    return np.array(mse2_app), np.array(ss2_app)

# reconstruction images, mse and ssim calculations
def recon_cVAE(dataloader):
    mse3_app, ss3_app = [], []
    data_app, recon_app = [], []
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            print('Step = ', step)
            data,mod = batch
            data = data.to(device1)
            mod = mod.to(device1)
            recon,_,_,_ = cvae15(data,mod)
           
            data = data.detach().cpu().numpy()
            recon = recon.detach().cpu().numpy()
            
            mse, ss = mse_ssim_samplewise(data,recon)
                        
            mse3_app.append(mse)
            ss3_app.append(ss)
            
            # save data and recon for plotting
            if (step == 0) or (step == 1) or (step == 2):
                data_app.append(data)
                recon_app.append(recon)
            
    return np.array(data_app), np.array(recon_app), np.array(mse3_app), np.array(ss3_app)

org_tr, recon_tr, mse_tr, ssim_tr = recon_cVAE(dataloader_trainval)
org_te, recon_te, mse_te, ssim_te = recon_cVAE(dataloader_test)

print("Original training shape",org_tr.shape)
print("Original test shape",org_te.shape)

print("ssim training shape",ssim_tr.shape)
print("ssim test shape",ssim_te.shape)

# change the shape of ssim and mse
ssim_tr = ssim_tr.reshape(ssim_tr.shape[0]*ssim_tr.shape[1],ssim_tr.shape[2])
ssim_te = ssim_te.reshape(ssim_te.shape[0]*ssim_te.shape[1],ssim_te.shape[2])
mse_tr = mse_tr.reshape(mse_tr.shape[0]*mse_tr.shape[1],mse_tr.shape[2])
mse_te = mse_te.reshape(mse_te.shape[0]*mse_te.shape[1],mse_te.shape[2])

# change the shape again
ssim_tr, ssim_te = ssim_tr.reshape(1400,48,15), ssim_te.reshape(100,48,15)
mse_tr, mse_te = mse_tr.reshape(1400,48,15), mse_te.reshape(100,48,15)

# reshape org_tr and recon_tr
org_tr = org_tr.reshape(org_tr.shape[0]*org_tr.shape[1],15,256,256)
org_te = org_te.reshape(org_te.shape[0]*org_te.shape[1],15,256,256)
recon_tr = recon_tr.reshape(recon_tr.shape[0]*recon_tr.shape[1],15,256,256)
recon_te = recon_te.reshape(recon_te.shape[0]*recon_te.shape[1],15,256,256)

# change the shape again
org_tr, org_te = org_tr.reshape(2,48,15,256,256), org_te.reshape(2,48,15,256,256)
recon_tr, recon_te = recon_tr.reshape(2,48,15,256,256), recon_te.reshape(2,48,15,256,256)

print("Original training shape",org_tr.shape)
print("Original test shape",org_te.shape)
print("ssim tr shape",ssim_tr.shape)
print("ssim te shape",ssim_te.shape)

#%% plot original and reconstruction images - 3 x 5 pictures
allprojnames = ['$x-p_x$','$x-y$','$x-p_y$','$x-z$','$x-p_z$',
                '$y-p_x$','$y-p_y$','$y-z$','$y-p_z$','$z-p_x$',
                '$z-p_y$','$z-p_z$','$p_x-p_y$','$p_x-p_z$','$p_z-p_y$']
    
org_te_un = UnNormalize(org_te, minval_te, maxval_te)
recon_te_un = UnNormalize(recon_te, minval_te, maxval_te)

org_te_un_log = np.log(1 + org_te_un)
recon_te_un_log = np.log(1 + recon_te_un)

randrun = 0
n_mod_recon = 39 # 0,23,47 for paper # 1,2,3,4,9,19,29,39 for supp. info
plot_samples = 15
plt.figure(figsize=(15,12))
# original images as 3 x 5
for p in range(15):
    cols = 5
    plt.subplot(int(plot_samples/cols) + 1, cols, p + 1)
    plt.suptitle('Original projections in module ' + str(n_mod_recon+1) , fontsize = 32, y = 0.99)
    plt.title(str(allprojnames[p]), fontsize = 28)
    plt.imshow(org_te_un_log[randrun,n_mod_recon,p,:,:], aspect='auto', origin='lower', cmap='plasma')
    plt.axis('OFF')
plt.tight_layout()
path = 'E:/MSR/codes/RFLA_VAE_LSTM_pyt/figures/fig_reconstruction/'
#plt.savefig(path + 'reconablity_orgimgs_log_mod_'+str(n_mod_recon+1) + '.png', bbox_inches="tight", dpi=300)

# reconstructed images as 3 x 5
plt.figure(figsize=(15,12))
for p in range(15):
    cols = 5
    plt.subplot(int(plot_samples/cols) + 1, cols, p + 1)
    plt.suptitle('Reconstruction projections in module ' + str(n_mod_recon+1) , fontsize = 32, y = 0.99)
    plt.title(str(allprojnames[p]), fontsize = 28)
    plt.imshow(recon_te_un_log[randrun,n_mod_recon,p,:,:], aspect='auto', origin='lower', cmap='plasma')
    plt.axis('OFF')
plt.tight_layout()
path = 'E:/MSR/codes/RFLA_VAE_LSTM_pyt/figures/fig_reconstruction/'
#plt.savefig(path + 'reconablity_reconimgs_log_mod_'+str(n_mod_recon+1) + '.png', bbox_inches="tight", dpi=300)

#%% plots for MSE and SSIM for the entire training and test with error bounds
def mu_sigma(data):
    mu = np.mean(data,axis=0)
    sigma = np.std(data,axis=0)
    return mu,sigma

mse_recon_avg_tr = np.mean(mse_tr, axis = 0)
ssim_recon_avg_tr = np.mean(ssim_tr, axis = 0)
mse_recon_avg_te = np.mean(mse_te, axis = 0)
ssim_recon_avg_te = np.mean(ssim_te, axis = 0)

print('Avg. MSE Recon te shape', mse_recon_avg_te.shape)
print('Avg. SSIM Recon te shape', ssim_recon_avg_te.shape)

mu_mse_recon_avg_tr, sigma_mse_recon_avg_tr = mu_sigma(mse_recon_avg_tr)
mu_ssim_recon_avg_tr, sigma_ssim_recon_avg_tr = mu_sigma(ssim_recon_avg_tr)

mu_mse_recon_avg_te, sigma_mse_recon_avg_te = mu_sigma(mse_recon_avg_te)
mu_ssim_recon_avg_te, sigma_ssim_recon_avg_te = mu_sigma(ssim_recon_avg_te)

print('Avg. MSE Recon mu te shape', mu_mse_recon_avg_te.shape)
print('Avg. SSIM Recon mu te shape', mu_ssim_recon_avg_te.shape)

lb_mse_tr = mu_mse_recon_avg_tr - sigma_mse_recon_avg_tr
ub_mse_tr =  mu_mse_recon_avg_tr + sigma_mse_recon_avg_tr
lb_mse_te = mu_mse_recon_avg_te - sigma_mse_recon_avg_te
ub_mse_te = mu_mse_recon_avg_te + sigma_mse_recon_avg_te

lb_ssim_tr = mu_ssim_recon_avg_tr - sigma_ssim_recon_avg_tr
ub_ssim_tr =  mu_ssim_recon_avg_tr + sigma_ssim_recon_avg_tr
lb_ssim_te = mu_ssim_recon_avg_te - sigma_ssim_recon_avg_te
ub_ssim_te = mu_ssim_recon_avg_te + sigma_ssim_recon_avg_te

plt.figure()
xaxis = np.arange(1,16)
fig, ax = plt.subplots(1, 2, figsize=(18, 3))
fig.suptitle('Average reconstruction MSE and SSIM', fontsize = 18)
ax[0].plot(np.arange(1,16),mu_mse_recon_avg_tr,'o-',c ='tab:blue')
ax[0].plot(np.arange(1,16),mu_mse_recon_avg_te,'^--',c='tab:blue')
ax[1].plot(np.arange(1,16),mu_ssim_recon_avg_tr,'o-',c = 'tab:orange')
ax[1].plot(np.arange(1,16),mu_ssim_recon_avg_te,'^--',c='tab:orange')

ax[0].set_ylabel('Avg. recon. MSE', fontsize = 18)
ax[1].set_ylabel('Avg. recon. SSIM', fontsize = 18)

ax[0].set_ylim([0.1e-7,10e-7])
ax[1].set_ylim([0.98,1.0])

ax[0].set_yticks([2e-7,4e-7,6e-7,8e-7,10e-7])
ax[1].set_yticks([0.98,0.985,0.99,0.995,1.00])

ax[0].set_yticklabels((['2','4','6','8','10e-7']),fontsize=15)
ax[1].set_yticklabels((['0.980','0.985','0.990','0.995','1.0']),fontsize=15)

ax[0].set_xticks(range(1,16))
ax[1].set_xticks(range(1,16))

ax[0].set_xticklabels((str(i) for i in range(1,16)),fontsize=15)
ax[1].set_xticklabels((str(i) for i in range(1,16)),fontsize=15)

ax[0].set_xlabel('6d LPS projections', fontsize = 18)
ax[1].set_xlabel('6d LPS projections', fontsize = 18)

ax[0].legend(['train','test'], fontsize=15)
ax[1].legend(['train','test'], fontsize=15)

plt.savefig('mse_ssim_recon.png', bbox_inches='tight' ,dpi=300)

#%% =====>>>> LATENT SPACE RESULTS <<<<======
############################################################################################
##################  LATENT SPACE VISUALIZATION PART ########################################
############################################################################################

#%% plot multidimensional latent space as different 2d projections
from lib.colorplots2d import COLORPLOTS_2D

def latentspace(dataloader,model):
    model.eval()
    latent_app = []
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            data, mod = batch
            data = data.to(device1)
            mod = mod.to(device1)
            _,_,_,z = model(data,mod)
            latent_app.append(z.cpu().detach())
        latent_app = torch.cat(latent_app)
    return latent_app

latent_train = latentspace(dataloader_train,cvae15).numpy()
latent_val = latentspace(dataloader_val,cvae15).numpy()
latent_combine = latentspace(dataloader_trainval,cvae15).numpy()
latent_test = latentspace(dataloader_test,cvae15).numpy()

print("Shape of the latent train", latent_train.shape)
print("Shape of the latent validation", latent_val.shape)
print("Shape of the latent combine", latent_combine.shape)
print("Shape of the latent test", latent_test.shape)

# color plots in latent
for i in range(1,8):
    COLORPLOTS_2D(latent_combine,imgs_all_tr.shape[0],0,i,'2d proj. of 8d latent space')
    #plt.savefig(path +'latent_parallelcoords_' + str(0) + '_' + str(i), dpi=300)
    
#%% parallel cords visualization of the 8d latent space
from lib.parallelcords1_plot import parallelcords1

# plot few training and validation samples in parralelcords
nlines1 = np.random.randint(0, latent_train.shape[0], 2000)
nlines2 = np.random.randint(0, latent_val.shape[0], 300)

data_train_pc = latent_train[nlines1]
data_val_pc = latent_val[nlines2]

# plotting train and test sets
N1 = np.ones(data_train_pc.shape[0],dtype='int64')
N2 = np.ones(data_val_pc.shape[0],dtype='int64')*2
coloring = np.concatenate((N1,N2),axis=0)
data_pc = np.concatenate((data_train_pc,data_val_pc),axis=0)

color1 = "tab:blue"
color2 = "tab:orange"

parallelcords1(data_pc,coloring,color1,color2)
#plt.savefig('latent_parallelcords.png', dpi=300)

#%% parallel cords visualization of the 8d latent space with different modules separated
from lib.parallelcords_modwise import PC_modwise

latent_combine_rs = latent_combine.reshape(1400,48,8)
n_lines = np.random.randint(0, latent_combine_rs.shape[0], 25) # 5 random lines for each module

coloring = []
data_pc = []
for i in range(48):
    data_mod = latent_combine_rs[n_lines, i, :]
    N = np.ones(data_mod.shape[0],dtype='int64')*(i+1)
    coloring.append(N)
    data_pc.append(data_mod)

data_pc = np.concatenate(data_pc, axis = 0)

PC_modwise(data_pc, n_lines.shape[0])
path = 'E:/MSR/codes/RFLA_VAE_LSTM_pyt/figures/fig_latentspace/'
plt.savefig(path + 'latent_parallelcords_modwise.png', dpi=300)

#%% latent space via PCA
from lib.PCA import PCAmodel
from lib.trajectors import TRAJECTORIESinLATENT
from lib.colorplotsinlatent import COLORPLOTSinLATENT

pca = PCAmodel()
nd = 2

(Xtr_pca, evr_tr_pca, recon_tr_pca) = pca.pcabuild(latent_train, nd)
(Xval_pca, evr_val_pca, recon_val_pca) = pca.pcabuild(latent_val, nd)
(Xc_pca, evr_tc_pca, recon_tc_pca) = pca.pcabuild(latent_combine, nd)

plt.figure(figsize=(6, 6))
plt.title('PCA of the latent space', fontsize = 20)
plt.scatter(Xtr_pca[:, 0], Xtr_pca[:, 1], c='tab:blue', marker=".")
plt.scatter(Xval_pca[:, 0], Xval_pca[:, 1], c='tab:orange', marker=".")
plt.legend(['Train','Test'], fontsize=18)
plt.xlabel('pca-1', fontsize=20)
plt.ylabel('pca-2', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
#plt.savefig('pca1_of_latent.png', dpi=300)   

# color plots in latent
COLORPLOTSinLATENT(Xc_pca,imgs_all_tr.shape[0], 'pca of 8d latent space', 'pca')
#plt.savefig(path + 'pca2_of_latent.png', dpi=300)   

modidx, prt = 4, 3
title = 'Trajectories in the latent manifold'
TRAJECTORIESinLATENT(Xc_pca,modidx,prt,title,'pca')
#plt.savefig('traj_pca2_of_latent.png', dpi=300)

#%% t-sne of the latent space
from sklearn.manifold import TSNE

size_train = latent_train.shape[0]
tsne = TSNE(n_components=2, learning_rate='auto',
            init ='pca').fit_transform(latent_combine)
X_train_tsne = tsne[0:size_train,:]
X_test_tsne  = tsne[size_train:,:]

plt.figure(figsize=(6, 6))
plt.title('t-SNE of the latent space', fontsize = 20)
plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c='tab:blue', marker=".")
plt.scatter(X_test_tsne[:, 0], X_test_tsne[:, 1], c='tab:orange', marker=".")
plt.legend(['Train', 'Test'], loc='best', fontsize=18)
plt.xlabel('tsne1', fontsize=20)
plt.ylabel('tsne2', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('tsne1_of_latent.png', dpi=300)   
plt.show()

# color plots in latent
COLORPLOTSinLATENT(tsne,imgs_all_tr.shape[0],'t-sne of 8d latent space', 'tsne')
plt.savefig(path + 'tsne2_of_latent.png', dpi=300)   

modidx, prt = 4, 3
title = 'Trajectories in the latent manifold'
TRAJECTORIESinLATENT(tsne,modidx,prt,title,'tsne')
#plt.savefig('traj_tsne2_of_latent.png', dpi=300)   

#%% UMAP visualization
import umap            
reducer = umap.UMAP(n_components=2)
mapper = reducer.fit(latent_combine)

COLORPLOTSinLATENT(mapper.embedding_,imgs_all_tr.shape[0], 'umap of 8d latent space', 'umap')
plt.savefig(path + 'umap_of_latent.png', dpi=150)

modidx, prt = 4, 3
title = 'Trajectories in the latent manifold'
TRAJECTORIESinLATENT(mapper.embedding_,modidx,prt,title,'umap')
#plt.savefig('traj_umap2_of_latent.png', dpi=300)

#%% =====>>>> START GENERATION PART <<<<======
############################################################################################
#########################  GENERATION PART #################################################
############################################################################################

#%% latent space sampler and decoding
n_mod = 29  # 0, 23 47 for paper # 1,2,3,4,9,19,29,39 for supplementary
n_mod_nm = mod_num /48
n_samples = 100
total_pertb = imgs_all_tr_rs_tensor.shape[0]

latent_combine_rs = latent_combine.reshape(imgs_all_tr.shape[0],imgs_all_tr.shape[1],latentdim)
latent_per_mod = latent_combine_rs[:,n_mod,:]

# based on perturbation within max-min bounds
ub_ls = np.max(latent_per_mod, axis = 0)
lb_ls = np.min(latent_per_mod, axis = 0)

# sample the latent space conditionally
def cvae_sampler_random(n_samples, lb_ls, ub_ls):
    rand_ls_app = []
    for i in range(latent_combine.shape[1]):
        rand_ls = lb_ls[i] + (ub_ls[i]-lb_ls[i])*np.random.random_sample((n_samples,))
        rand_ls_app.append(rand_ls)
    all_rand_ls = np.array(rand_ls_app)
    rand_ls_trans = np.transpose(all_rand_ls)
    rand_ls_32 = rand_ls_trans.astype("float32")
    #print(rand_ls_32.shape)
    rand_ls_32 = torch.tensor(rand_ls_32).to(device1)
    with torch.no_grad():
        cvae15.to(device1)
        cvae15.eval()
        generate_samples = cvae15.decode(rand_ls_32)
    
    return (generate_samples.detach().cpu().numpy(), rand_ls_32.detach().cpu().numpy())

generatedsamples, ls_gensamples = cvae_sampler_random(n_samples, lb_ls, ub_ls)

#%% Generated samples in parallal corrds and 2d PCA space
# parallel coords plot for particular modules
from lib.parallelcords2_plot import parallelcords2
from lib.PCA2 import PCAbuild

all_colors = ["gray","k","tab:blue","tab:green","tab:red","tab:orange","tab:pink","tab:cyan"]

if (n_mod == 0):
    bgcolor = all_colors[2]
elif (n_mod == 23):
    bgcolor = all_colors[3]
elif (n_mod == 47):
    bgcolor = all_colors[4]
else:
    bgcolor = all_colors[5]

def create_reflection_matrix(original_matrix):
    dim = original_matrix.shape[1]
    reflection_matrix = np.eye(dim)
    reflection_matrix[1, 1] = -1  # Negate the first element
    reflected_matrix = np.dot(original_matrix, reflection_matrix)
    return reflected_matrix

mean_X, projection_matrix = PCAbuild(latent_combine, 2)
centered_X = latent_combine - mean_X
X_projected = centered_X.dot(projection_matrix)

# X_projected gives a x-axis reflected matrix as compared to sklearn version of PCA.
Xc_pca = create_reflection_matrix(X_projected)
Xc_pca_rs = Xc_pca.reshape(1400,48,2)
Xc_pca_mod = Xc_pca_rs[:,n_mod,:]

# use the same projection matrix to see PCA for new generated samples
latent_gen_centered = ls_gensamples - mean_X
latent_gen_projected = latent_gen_centered.dot(projection_matrix)
latent_gen_pca = create_reflection_matrix(latent_gen_projected)

plt.figure(figsize=(6, 6))
plt.title('PCA of 8d latent space for module '+str(n_mod+1), fontsize = 20)
plt.scatter(Xc_pca[:, 0], Xc_pca[:, 1], c=all_colors[0], 
            marker=".", label = 'PCA of latent - all')
plt.scatter(Xc_pca_mod[:, 0], Xc_pca_mod[:, 1], c = bgcolor, 
            marker=".", label = 'PCA of latent'+' - mod. '+str(n_mod+1))
plt.scatter(latent_gen_pca[:, 0], latent_gen_pca[:, 1], c=all_colors[1], 
            marker="*", 
            label = 'PCA of sampled points')
plt.xlabel('pca-1', fontsize=20)
plt.ylabel('pca-2', fontsize=20)
plt.legend(fontsize=15)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.show()

#%% plot the points which are inside the region
# - Some of the points are scattered in the vicinity of the region due to the
# information loss in dimensionality reduction.
# We are showing 5 generated images, we will take points inside the region

from matplotlib.path import Path
import random 

def set2_inside_set1(set1,set2):
    # set-1 is defined by points of the module and set-2 are the generated points
    path = Path(set1)
    selected_indices = [i for i, (x, y) in enumerate(set2) if path.contains_point((x, y),radius = 0.0)]
    selected_points = [(x, y) for x, y in set2[selected_indices]]
    
    return (np.array(selected_indices), np.array(selected_points))

idx_gen_inside, latent_gen_pca_inside = set2_inside_set1(Xc_pca_mod, latent_gen_pca)
ls_gensamples_inside = ls_gensamples[idx_gen_inside,:]

print("Number of points inside the region = ", idx_gen_inside.shape[0])

# select few samples to plot
n_samples_plt = 5
idx_gen_inside_plt = random.sample(range(0 , idx_gen_inside.shape[0]), n_samples_plt)
latent_gen_pca_plt = latent_gen_pca_inside[idx_gen_inside_plt,:]
ls_gensamples_plt = ls_gensamples_inside[idx_gen_inside_plt]

# parallecorrds
path = 'E:/MSR/codes/RFLA_VAE_LSTM_pyt/figures/fig_generation/'
N1 = np.ones(latent_per_mod.shape[0],dtype='int64')
N2 = np.ones(ls_gensamples_plt.shape[0],dtype='int64')*2
coloring = np.concatenate((N1, N2),axis=0) 
Nc = np.concatenate((latent_per_mod, ls_gensamples_plt),axis=0)
parallelcords2(Nc, coloring, bgcolor, all_colors[1], n_mod)
plt.savefig(path+'latent_parallelcords_gen_'+'mod_'+str(n_mod+1)+'.png', dpi=300)

# pca
plt.figure(figsize=(6, 6))
plt.title('b) PCA of latent space for mod. '+str(n_mod+1), fontsize = 22)
plt.scatter(Xc_pca[:, 0], Xc_pca[:, 1], c=all_colors[0], 
            marker=".", label = 'PCA of latent - all')
plt.scatter(Xc_pca_mod[:, 0], Xc_pca_mod[:, 1], c = bgcolor, alpha = 0.3, 
            marker=".", label = 'PCA of latent'+' - mod. '+str(n_mod+1))
plt.scatter(latent_gen_pca_plt[:, 0], latent_gen_pca_plt[:, 1], c=all_colors[1], 
            marker="*", 
            label = 'PCA of sampled points')
plt.xlabel('pca-1', fontsize=20)
plt.ylabel('pca-2', fontsize=20)
plt.legend(fontsize=15)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig(path+'gensamplepca_mod_'+str(n_mod+1)+'.png', dpi=300)
plt.show()

#%% generate the sample using deocder == all projections
allprojnames = ['$x-p_x$','$x-y$','$x-p_y$','$x-z$','$x-p_z$',
                '$y-p_x$','$y-p_y$','$y-z$','$y-p_z$','$z-p_x$',
                '$z-p_y$','$z-p_z$','$p_z-p_y$','$p_x-p_z$','$p_z-p_y$']

imgs_all_tr_nm = imgs_all_tr_rs_tensor.reshape(1400,48,15,256,256)
randrun1 = np.random.randint(0, imgs_all_tr.shape[0], 1)
realimages_plt = imgs_all_tr_nm[randrun1,n_mod,:,:,:].numpy()
generatedsamples_plt = generatedsamples[idx_gen_inside_plt]

# Unnormalize the data and log
realimages_plt_un = UnNormalize(realimages_plt, minval_tr, maxval_tr)
generatedsamples_plt_un = UnNormalize(generatedsamples_plt, minval_tr, maxval_tr)

realimages_plt_un_log = np.log(1 + realimages_plt_un)
generatedsamples_plt_un_log = np.log(1 + generatedsamples_plt_un)

# plot generated images
plot_samples = 15
plt.figure()
fig, ax = plt.subplots(6, plot_samples, figsize=(25, 9))
fig.subplots_adjust(hspace=0.25)
fig.subplots_adjust(wspace=0.25)
fig.text(0.4, 0.95,'c) Generated projections for module '+ str(n_mod+1), va='center', 
         rotation='horizontal',fontsize=20)
fig.text(0.11, 0.18,'gen 5', va='center', rotation='vertical',fontsize=16)
fig.text(0.11, 0.30,'gen 4', va='center', rotation='vertical',fontsize=16)
fig.text(0.11, 0.43,'gen 3', va='center', rotation='vertical',fontsize=16)
fig.text(0.11, 0.56,'gen 2', va='center', rotation='vertical',fontsize=16)
fig.text(0.11, 0.69,'gen 1', va='center', rotation='vertical',fontsize=16)
fig.text(0.11, 0.82,'org.', va='center', rotation='vertical',fontsize=18)

for p in range(plot_samples):
    ax[0, p].set_title(str(allprojnames[p]), fontsize = 18)
    ax[0, p].imshow(realimages_plt_un_log[0,p,:,:],aspect='auto', origin='lower', cmap='plasma')
    ax[0, p].axis('OFF')
    for i in range(1,6):
        gen = generatedsamples_plt_un_log[i-1,p,:,:]
        ax[i, p].imshow(gen, aspect='auto', origin='lower', cmap='plasma')
        ax[i, p].axis('OFF')
plt.savefig(path + 'genimage_all_projs_mod_'+str(n_mod+1)+'.png', dpi=600)
plt.show()

#%% calculate mse and ssim for generated images
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

randrun2 = np.random.randint(0,imgs_all_tr.shape[0], n_samples)
realimages_cal = imgs_all_tr_nm[randrun2,n_mod,:,:,:].numpy()

mse_gen_app, ssim_gen_app = [], []
for i in range(generatedsamples.shape[0]):
    for j in range(15):
        real = realimages_cal[i,j,:,:]
        gen = generatedsamples[i,j,:,:]
        mse_gen = mean_squared_error(real, gen)
        ssim_gen = ssim(real, gen, data_range = real.max() - real.min())
        
        mse_gen_app.append(mse_gen)
        ssim_gen_app.append(ssim_gen)

mse_gen_app = np.array(mse_gen_app)
ssim_gen_app = np.array(ssim_gen_app)
print("Shape of mse gen", mse_gen_app.shape)

mse_gen_mean = np.mean(mse_gen_app)
ssim_gen_mean = np.mean(ssim_gen_app)
print("MSE gen mean", mse_gen_mean)
print("SSIM gen mean", ssim_gen_mean)

#% upload the the resnet50_model classifier
from lib.pretrainedresnet50 import resnet50_evalgenerator
generatedsamples_torch = torch.from_numpy(generatedsamples)

path = 'E:/MSR/codes/RFLA_VAE_LSTM_pyt/weights/'
with torch.no_grad():
    resnet50_model = resnet50_evalgenerator(device1, path)
    resnet50_model.eval()
    #print('Model architecture', resnet50_model)
    # calculate accuracy of the generated samples usinf pretrained resnet50_model
    y_pred_gen = resnet50_model(generatedsamples_torch.to(device1))
    _, y_predgen_class = torch.max(y_pred_gen, 1)
    y_predgen_class = y_predgen_class.detach().cpu().numpy()

print('The true class = ', n_mod)
print('The predicted class = ', y_predgen_class)

n_correct_loc = np.where(y_predgen_class == n_mod)[0]
TP_gen = len(n_correct_loc)
FP_gen = n_samples - TP_gen
Precision = TP_gen/(TP_gen+FP_gen)

del y_pred_gen, resnet50_model, y_predgen_class
torch.cuda.empty_cache()

#%% use resnet50 to calculate fid score
from lib.pretrainedresnet50 import resnet50_evalgenerator
from lib.fid_torch import calculate_fid

dev = device1

randrun1 = np.random.randint(0,imgs_all_tr.shape[0], n_samples)
imgs_all_tr_nm = imgs_all_tr_rs_tensor.reshape(1400,48,15,256,256)
real_imgs = imgs_all_tr_nm[randrun1,n_mod,:,:,:].numpy()

def get_features_after_pooling_layer(dev):
    with torch.no_grad():
        resnet50 = resnet50_evalgenerator(dev, path)
        resnet50.eval()
        # Find the global average pooling layer
        features_after_pooling = list(resnet50.children())[:-1]  # Remove the final fully connected layer
        del resnet50
        torch.cuda.empty_cache()
    return torch.nn.Sequential(*features_after_pooling)

features_after_pooling = get_features_after_pooling_layer(dev)
features_realimgs = features_after_pooling(torch.from_numpy(real_imgs).to(dev))
features_genimgs = features_after_pooling(generatedsamples_torch.to(dev))

# reduce the shape to (samples,features)
features_real = features_realimgs.reshape(features_realimgs.shape[0],
                                          features_realimgs.shape[1]*features_realimgs.shape[2]*
                                          features_realimgs.shape[3])

features_gen = features_genimgs.reshape(features_genimgs.shape[0],
                                        features_genimgs.shape[1]*features_genimgs.shape[2]*
                                        features_genimgs.shape[3])

del features_realimgs, features_genimgs
torch.cuda.empty_cache()

print("Shape of Resnet50 features (Real images)", features_real.shape)
print("Shape of Resnet50 features (Generated images)", features_gen.shape)

features_real = features_real.detach().cpu()
features_gen = features_gen.detach().cpu()

fidscores = calculate_fid(features_real,features_gen)
print("Fid score is", fidscores)

#%% calculate acc, precision, recall and confusion metrics for all classes at once
from sklearn.metrics import recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
def cvae_sample_random_allmodules(n_samples):
    gensamples_allclass = []
    for m in range(48):
        n_mod = m
        
        latent_combine_rs = latent_combine.reshape(imgs_all_tr.shape[0],imgs_all_tr.shape[1],latentdim)
        latent_per_mod = latent_combine_rs[:,n_mod,:]
    
        # based on perturbation within max-min bounds
        ub_ls = np.max(latent_per_mod, axis = 0)
        lb_ls = np.min(latent_per_mod, axis = 0)
        
        # random sampling and collect generated images
        rand_ls_app = []
        for i in range(latent_combine.shape[1]):
            rand_ls = lb_ls[i] + (ub_ls[i]-lb_ls[i])*np.random.random_sample((n_samples,))
            rand_ls_app.append(rand_ls)
        all_rand_ls = np.array(rand_ls_app)
        rand_ls_trans = np.transpose(all_rand_ls)
        rand_ls_32 = rand_ls_trans.astype("float32")
        generate_samples = cvae15.decode(torch.tensor(rand_ls_32).to(device1))
        gensamples_allclass.append(generate_samples.detach().cpu().numpy())
        
    y_true = np.repeat(np.arange(0,48,1),n_samples)
    return (np.array(gensamples_allclass), y_true)

    
def classificationmetrics_forgenerator(y_true, gen_images):
    with torch.no_grad():
        resnet50 = resnet50_evalgenerator(device1,path)
        resnet50.eval()
    
        y_predgen_class_app = []
        for i in range(gen_images.shape[0]): # loop for individual image to save memory
            gen = gen_images[i]
            #print(gen.shape)
            gen = torch.unsqueeze(gen, 0) # add dim at 0 axis
            y_pred_gen = resnet50(gen)
            _, y_predgen_class = torch.max(y_pred_gen, 1)
            y_predgen_class = y_predgen_class.detach().cpu().numpy()
            y_predgen_class_app.append(y_predgen_class)
            del gen, y_pred_gen # to save gpu memory
        
    y_predgen_class_app = np.array(y_predgen_class_app)
    precision = precision_score(y_true, y_predgen_class_app, average = None)
    recall = recall_score(y_true, y_predgen_class_app, average = None)
    confusion = confusion_matrix(y_true, y_predgen_class_app)
    
    del resnet50
    torch.cuda.empty_cache()
    return (y_predgen_class, precision, recall, confusion)
  
gensamples_allclass, y_true = cvae_sample_random_allmodules(n_samples)
gensamples_allclass_rs = gensamples_allclass.reshape(48*n_samples,15,256,256)
gensamples_allclass_rs = torch.from_numpy(gensamples_allclass_rs).to(device1)

y_predgen_class, precision, recall, confusion =  classificationmetrics_forgenerator(y_true, gensamples_allclass_rs)

print('Precison = ', precision)
print('Recall = ', recall)
cmp = ConfusionMatrixDisplay(confusion, display_labels=np.arange(48))
fig, ax = plt.subplots(figsize=(20,20))
cmp.plot(ax=ax)

# torch.cuda.empty_cache()
# precision and recall for particular module
prec_mod = precision[n_mod]
recall_mod = recall[n_mod]

print('MSE for module ' + str(n_mod+1) + ' projections ===>>>', mse_gen_mean)
print('SSIM for module ' + str(n_mod+1) + ' projections ===>>>', ssim_gen_mean)
print('FID for module ' + str(n_mod+1) + ' projections ===>>>', fidscores)
print('Precison for module ' + str(n_mod+1) + ' projections ===>>>', prec_mod)
print('Recall for module ' + str(n_mod+1) + ' projections ===>>>', recall_mod)

del gensamples_allclass_rs
torch.cuda.empty_cache()
    
#%% =====>>>> START LSTM PART <<<<======
############################################################################################
#########################  FORECASTING PART #################################################
############################################################################################

#%% transform the dataset from (4800,9) to (100,48,9) for rnn
# transform the dataset from (x,9) to (x/48,48,9) 
latent_combine_rs = np.zeros((imgs_all_tr.shape[0],imgs_all_tr.shape[1],
                              latent_combine.shape[1]))
for i in range(imgs_all_tr.shape[0]):
    idx1 = 0+imgs_all_tr.shape[1]*i
    idx2 = imgs_all_tr.shape[1]+imgs_all_tr.shape[1]*i
    latent_combine_rs[i,:,:] = latent_combine[idx1:idx2,:]

# test set
latent_test_rs = np.zeros((imgs_all_te.shape[0],imgs_all_te.shape[1],
                              latent_combine.shape[1]))
for i in range(imgs_all_te.shape[0]):
    idx1 = 0+imgs_all_te.shape[1]*i
    idx2 = imgs_all_te.shape[1]+imgs_all_te.shape[1]*i
    latent_test_rs[i,:,:] = latent_test[idx1:idx2,:]
    
# split the dataset into training and validation
tr_lc_rs = latent_combine_rs[0:500,:,:]
val_lc_rs = latent_combine_rs[500:latent_combine_rs.shape[0],:,:]
test_lc_rs = latent_test_rs

# split the dataset into different chunks for supervised training
steps = np.arange(2,imgs_all_tr.shape[1])
steps_list = steps.tolist()

length_tr = tr_lc_rs.shape[0]
length_val = val_lc_rs.shape[0]
length_test = test_lc_rs.shape[0]

# split the dataset into different chunks of variable sizes
from lib.split_dataset import SPLIT_DATASET

(Xtrain_rnn,Ytrain_rnn) = SPLIT_DATASET(tr_lc_rs,steps,length_tr)
(Xval_rnn,Yval_rnn) = SPLIT_DATASET(val_lc_rs,steps,length_val)
(Xtest_rnn,Ytest_rnn) = SPLIT_DATASET(test_lc_rs,steps,length_test)

# pad the splitted dataset with zeros
from lib.pad_sequences_post import pad_dataset

padded_X_tr, all_len_tr = pad_dataset(Xtrain_rnn,steps,47,0)
padded_X_val, all_len_val = pad_dataset(Xval_rnn,steps,47,0)
padded_X_test, all_len_test = pad_dataset(Xtest_rnn,steps,47,0)

padded_X_tr_tensor = torch.from_numpy(padded_X_tr)
padded_X_val_tensor = torch.from_numpy(padded_X_val)
padded_X_test_tensor = torch.from_numpy(padded_X_test)

padded_Y_tr = torch.from_numpy(np.concatenate(Ytrain_rnn,axis=0))
padded_Y_tr_tensor = padded_Y_tr.clone().to(torch.float32)
padded_Y_val = torch.from_numpy(np.concatenate(Yval_rnn,axis=0))
padded_Y_val_tensor = padded_Y_val.clone().to(torch.float32)
padded_Y_test = torch.from_numpy(np.concatenate(Ytest_rnn,axis=0))
padded_Y_test_tensor = padded_Y_test.clone().to(torch.float32)

print("Shape of train dataset for RNN (Input)",padded_X_tr_tensor.shape)
print("Shape of train dataset for RNN (Labels)",padded_Y_tr_tensor.shape)
print("Shape of val dataset for RNN (Input)",padded_X_val_tensor.shape)
print("Shape of val dataset for RNN (Labels)",padded_Y_val_tensor.shape)
print("Shape of test dataset for RNN (Input)",padded_X_test_tensor.shape)
print("Shape of test dataset for RNN (Labels)",padded_Y_test_tensor.shape)

del padded_X_tr, padded_X_val, padded_X_test
del padded_Y_tr, padded_Y_val, padded_Y_test

#%% create dataset and dataloader
class DatasetforDataloader2(Dataset):
    def __init__(self,X,y,L):
        self.X = X
        self.y = y
        self.L = L
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,i):
        # create a tuple
        return self.X[i], self.y[i], self.L[i]
    
tr_dataset = DatasetforDataloader2(padded_X_tr_tensor,padded_Y_tr_tensor,all_len_tr)
valid_dataset = DatasetforDataloader2(padded_X_val_tensor,padded_Y_val_tensor,all_len_val)
test_dataset = DatasetforDataloader2(padded_X_test_tensor,padded_Y_test_tensor,all_len_val)

train_loader_rnn = torch.utils.data.DataLoader(dataset = tr_dataset,
                                               batch_size = 8,
                                               shuffle=False)

val_loader_rnn = torch.utils.data.DataLoader(dataset = valid_dataset,
                                             batch_size = 8,
                                             shuffle=False)

test_loader_rnn = torch.utils.data.DataLoader(dataset = test_dataset,
                                             batch_size = 8,
                                             shuffle=False)

#%% RNN model
import torch.nn as nn
from lib.rnn_varinpt import RNN
 
input_size = padded_X_tr_tensor.shape[2]  # features
sequence_length = padded_X_tr_tensor.shape[1] 
num_classes = padded_Y_tr_tensor.shape[1]
num_layers = 2
hidden_size = 64

device = device2
rnn = RNN(input_size, hidden_size, num_layers, num_classes)
#rnn = torch.nn.DataParallel(rnn)
rnn = rnn.to(device)

# Loss and optimizer
criterion_rnn = nn.MSELoss()
optimizer_rnn = torch.optim.Adam(rnn.parameters(), lr = 0.001)

#%% Train the RNN model
num_epochs = 200

def train_epoch_rnn(model,device,dataloader,optimizer,criterion_rnn):
    model.train() # Set train mode for model
    train_loss = []
    for step, batch in enumerate(dataloader):
        imgs, labels, length = batch
        imgs = imgs.clone().to(torch.float32)
        imgs = imgs.to(device)
        labels = labels.to(device)
        # forward pass
        outputs = model(imgs,length,device)
        loss = criterion_rnn(outputs, labels)
        # backward and optimizer
        optimizer_rnn.zero_grad()
        loss.backward()
        optimizer_rnn.step()
        # collect all loss
        train_loss.append(loss.item())
    return np.mean(train_loss)
        
def test_epoch_rnn(model,device,dataloader,criterion_rnn):
    model.eval() # Set the eval mode for model 
    test_loss = []
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            imgs, labels, length = batch
            imgs = imgs.clone().to(torch.float32)
            imgs = imgs.to(device)
            labels = labels.to(device)
            # prediction
            outputs = model(imgs,length,device)
            loss = criterion_rnn(outputs, labels)
            # collect all loss
            test_loss.append(loss.item())
    return np.mean(test_loss)

diz_loss_rnn = {'train_loss':[],'val_loss':[]}

for epoch in range(num_epochs):
   train_loss = train_epoch_rnn(rnn,device,train_loader_rnn,optimizer_rnn,criterion_rnn)
   val_loss = test_epoch_rnn(rnn,device,val_loader_rnn,criterion_rnn)
   print('\n EPOCH {}/{} \t train loss {} \t val loss {}'
         .format(epoch + 1, num_epochs,train_loss,val_loss))
   diz_loss_rnn['train_loss'].append(train_loss)
   diz_loss_rnn['val_loss'].append(val_loss)

# Save the model
torch.save(rnn.state_dict(), 'rnn_for_cvae15_model.pth') 

#%% load the model
rnn.load_state_dict(torch.load('rnn_for_cvae15_model.pth'))
rnn.eval()

#%% plots RNN training and test losses
plt.figure(figsize=(10,8))
plt.plot(diz_loss_rnn['train_loss'], '-ob', label='Train',)
plt.plot(diz_loss_rnn['val_loss'], '-or', label='Valid')
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Average Loss', fontsize=20)
plt.legend(["tr_total", "val_total"],fontsize=20)
plt.title('RNN loss',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

#%% single step prediction of trained rnn on validation set
past1 = 0
past2 = 4
nfeatures = val_lc_rs.shape[2]

with torch.no_grad():    
    xpast_ls = torch.from_numpy(val_lc_rs[0,past1:past2,:])
    xpast_ls = xpast_ls.reshape((1, past2-past1, nfeatures)).to(device)
    xpast_ls = xpast_ls.clone().to(torch.float32)
    length_ls = torch.tensor([xpast_ls.shape[1]])
    forecast_ls = rnn(xpast_ls,length_ls,device)

print("Forecasted latent sample",forecast_ls)
print("Original latent sample",val_lc_rs[0,past2+1,:])

# passing forecasted latent sample to generate images using trained decoder
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

with torch.no_grad():
    cvae15 = cvae15.to(device)
    forecast_img = cvae15.decode(forecast_ls).detach().cpu().numpy()
    forecast_img = np.squeeze(forecast_img,0)

projec = 11
forecast_img = forecast_img[projec,:,:]
real_img = imgs_all_tr_nm[0,past2,projec,:,:].numpy()
absdiff = np.abs(real_img-forecast_img)

mse = (real_img,forecast_img)
ssim_1 = ssim(real_img, forecast_img, data_range=forecast_img.max() - forecast_img.min())

print("MSE", mse)
print("SSIM", ssim_1)

maxval = np.max(real_img)
fig = plt.figure(figsize = (15,5)) 
fig.add_subplot(131)
plt.title("Original Sample at mod."+str(past2+1), fontsize = 20)
plt.imshow(real_img, aspect='auto', origin='lower', cmap='plasma',vmin=0,vmax=maxval)
plt.colorbar()
plt.tight_layout()
fig.add_subplot(132)
plt.title("Generated Sample at mod."+str(past2+1), fontsize = 20)
plt.imshow(forecast_img, aspect='auto', origin='lower', cmap='plasma',vmin=0,vmax=maxval)
plt.colorbar()
plt.tight_layout()
fig.add_subplot(133)
plt.title("Abs diff. at mod. "+str(past2+1), fontsize = 20)
plt.imshow(absdiff, aspect='auto', origin='lower', cmap='plasma')
#plt.text(10,230,"mse="+str(mse)+'\n'+'ssim='+str(ssim_1),c='w',fontsize=14)
plt.colorbar()
plt.tight_layout()

#%% multi-step prediction on the test set
from lib.recursive_rnn_pyt import RECURSION
past1 = 0
past2 = 4
xpast_ls = torch.from_numpy(test_lc_rs[0,past1:past2,:])
xpast_ls = xpast_ls.reshape((1, past2-past1, nfeatures)).to(device)
xpast_ls = xpast_ls.clone().to(torch.float32)
length_ls = torch.tensor([xpast_ls.shape[1]])

nfuture = 48-(past1+past2)
ls_all, decoded_all = RECURSION(rnn, cvae15, xpast_ls, length_ls, device, past1, past2, nfuture)
gen_all = np.array(decoded_all)

projec = 12 # 1,11,12
gen_all = gen_all[:,projec,:,:]
real_all = imgs_all_tr_nm[0,past2:nfuture+past2,projec,:,:] .numpy()
mse_all = reconstruction_error(gen_all, real_all)

print("All gen images shape",gen_all.shape)
print("All real images shape",real_all.shape)

#path_1 = 'E:/MSR/codes/RFLA_VAE_LSTM_pyt/figures/fig_forecasting/animation_1/'
path_2 = 'E:/MSR/codes/RFLA_VAE_LSTM_pyt/figures/fig_forecasting/'

# prediction plots from tcVAE on test set
ssim_all = []
for i in range(gen_all.shape[0]):
    real = real_all[i,:,:]
    gen = gen_all[i,:,:]
    diff = np.abs(real - gen)
    mse = mse_all[i]
    mse = format(float(mse),'.2e')
    
    maxval_1 = np.max(real)
    
    ssim_3 = ssim(real, gen, data_range = real.max() - real.min())
    ssim_3 = np.round(ssim_3,4)
    ssim_all.append(ssim_3)
    
    real_un = UnNormalize(real, minval_te, maxval_te)
    gen_un = UnNormalize(gen, minval_te, maxval_te)
    diff_un = np.abs(real_un - gen_un)
    
    maxval_2 = np.max(real_un)
    
    real_un_log = np.log(1+UnNormalize(real, minval_te, maxval_te))
    gen_un_log = np.log(1+UnNormalize(gen, minval_te, maxval_te))
    diff_un_log = np.abs(real_un_log - gen_un_log)
    
    maxval_3 = np.max(real_un_log)
    
    maxval = maxval_3
    
    fig = plt.figure(figsize = (15, 5))
    fig.add_subplot(131)
    plt.title("Original at mod."+str(past2+i+1), fontsize = 28)
    plt.imshow(real_un_log, aspect='auto', origin='lower', cmap='plasma', vmin = 0, vmax = maxval)
    #plt.colorbar()
    plt.axis('OFF')
    plt.tight_layout()
    fig.add_subplot(132)
    plt.title("Forecasted at mod."+str(past2+i+1), fontsize = 28)
    plt.imshow(gen_un_log, aspect='auto', origin='lower', cmap='plasma', vmin = 0, vmax = maxval)
    #plt.colorbar()
    plt.axis('OFF')
    plt.tight_layout()
    fig.add_subplot(133)
    plt.imshow(diff_un_log, aspect='auto', origin='lower', cmap='plasma', vmin = 0, vmax = maxval)
    plt.text(10,215,"mse="+str(mse)+'\n'+'ssim='+str(ssim_3),c='w',fontsize=22)
    plt.title("Abs. Diff. at mod. "+str(past2+i+1), fontsize = 28)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20) 
    plt.tight_layout()
    plt.axis('OFF')
    #plt.savefig(path_1 + 'proj_' + str(projec) + '_log' +'_predmod_'+ str(past2+i+1).zfill(2) + '.png', dpi=300)
    plt.savefig(path_2 + 'proj_' + str(projec) + '_log' +'_predmod_'+ str(past2+i+1).zfill(2) + '.png', dpi=300)
    plt.show()

#%% calculate mse and ssim between forecasted and original for the TEST set
def ssim_dataset(realdata,gendata):
    ssim_app = []
    for i in range(realdata.shape[0]):
        real = realdata[i]
        gen = gendata[i]
        sim = ssim(real, gen, data_range=real.max() - real.min())
        ssim_app.append(sim)
    return np.array(ssim_app)

past1 = 0
past2 = 4
mse_app = []
ssim_app = []

for i in range(test_lc_rs.shape[0]):
    xpast_ls = torch.from_numpy(test_lc_rs[i,past1:past2,:])
    xpast_ls = xpast_ls.reshape((1, past2-past1, nfeatures)).to(device)
    xpast_ls = xpast_ls.clone().to(torch.float32)
    length_ls = torch.tensor([xpast_ls.shape[1]])
    
    nfuture = 48-(past1+past2)
    ls_all, decoded_all = RECURSION(rnn, cvae15, xpast_ls, length_ls, device, past1, past2, nfuture)
    gen_all = np.array(decoded_all)
    ls_all = np.array
    
    # only 11th projection
    projec = 11
    gen_all = gen_all[:,projec,:,:]
    
    # mse on the 11th projection
    real_all = imgs_all_tr_nm[i,past2:nfuture+past2,projec,:,:].numpy() 
    mse_all = reconstruction_error(gen_all, real_all)
    real_all = np.array(real_all)
    mse_app.append(mse_all)
    
    # ssim on the 11th projection
    ssim_all = ssim_dataset(gen_all, real_all)
    ssim_app.append(ssim_all)
    
mse_app = np.array(mse_app)
ssim_app = np.array(ssim_app)
    
print("All mse shape",mse_app.shape)
print("All ssim shape",ssim_app.shape)

mean_mse = np.mean(mse_app, axis=0)
std_mse = np.std(mse_app, axis=0)
mean_ssim = np.mean(ssim_app, axis=0)
std_ssim = np.std(ssim_app, axis=0)

path = 'E:/MSR/codes/RFLA_VAE_LSTM_pyt/figures/fig_forecasting/'
xaxis = np.arange(5,nfuture+5)
plt.figure(figsize = (30, 6))
plt.plot(xaxis, mean_mse, 'tab:orange')
plt.fill_between(xaxis, mean_mse-std_mse, mean_mse+std_mse,
                 alpha=0.2, facecolor='tab:orange',linewidth=4)
plt.xlabel("Module number (future states)",fontsize=20)
plt.ylabel("MSE",fontsize=20)
plt.title("Prediction: MSE on the test set",fontsize=20)
plt.xticks(xaxis,fontsize=20)
plt.yticks(fontsize=20)
plt.savefig(path + 'msepred_test.png', dpi=300)
plt.show()

plt.figure(figsize = (30, 6))
plt.plot(xaxis, mean_ssim, 'tab:orange')
plt.fill_between(xaxis, mean_ssim-std_ssim, mean_ssim+std_ssim,
                 alpha=0.2, facecolor='tab:orange',linewidth=4)
plt.xlabel("Module number (future states)",fontsize=20)
plt.ylabel("SSIM",fontsize=20)
plt.title("Prediction: SSIM on the test set",fontsize=20)
plt.xticks(xaxis,fontsize=20)
plt.savefig(path + 'ssimpred_test.png', dpi=300)
plt.yticks(fontsize=20)

#%% calculate mse and ssim between forecasted and original for the TRAIN set
past1 = 0
past2 = 4
mse_tr_app = []
ssim_tr_app = []

for i in range(latent_combine_rs.shape[0]):
    xpast_ls = torch.from_numpy(latent_combine_rs[i,past1:past2,:])
    xpast_ls = xpast_ls.reshape((1, past2-past1, nfeatures)).to(device)
    xpast_ls = xpast_ls.clone().to(torch.float32)
    length_ls = torch.tensor([xpast_ls.shape[1]])
    
    nfuture = 48-(past1+past2)
    ls_all, decoded_all = RECURSION(rnn, cvae15, xpast_ls, length_ls, device, past1, past2, nfuture)
    gen_all = np.array(decoded_all)
    
    # only 11th projection
    projec = 11
    gen_all = gen_all[:,projec,:,:]
    
    # mse on the 11th projection
    real_all = imgs_all_tr_nm[i,past2:nfuture+past2,projec,:,:].numpy()
    mse_tr_all = reconstruction_error(gen_all, real_all)
    real_all = np.array(real_all)
    mse_tr_app.append(mse_tr_all)
    
    # ssim on the 11th projection
    ssim_tr_all = ssim_dataset(gen_all, real_all)
    ssim_tr_app.append(ssim_tr_all)
    
mse_tr_app = np.array(mse_tr_app)
ssim_tr_app = np.array(ssim_tr_app)
    
print("All mse shape",mse_tr_app.shape)
print("All ssim shape",ssim_tr_app.shape)

mean_mse_tr = np.mean(mse_tr_app, axis=0)
std_mse_tr = np.std(mse_tr_app, axis=0)
mean_ssim_tr = np.mean(ssim_tr_app, axis=0)
std_ssim_tr = np.std(ssim_tr_app, axis=0)

xaxis = np.arange(5,nfuture+5)
plt.figure(figsize = (30, 6))
plt.plot(xaxis, mean_mse_tr, 'tab:blue')
plt.fill_between(xaxis, mean_mse_tr-std_mse_tr, mean_mse_tr+std_mse_tr,
                 alpha=0.2, facecolor='tab:blue',linewidth=4)
plt.xlabel("Module number (future states)",fontsize=20)
plt.ylabel("MSE",fontsize=20)
plt.title("Prediction: MSE on the training set",fontsize=20)
plt.xticks(xaxis,fontsize=20)
plt.yticks(fontsize=20)
plt.savefig(path + 'msepred_train.png', dpi=300)
plt.show()

plt.figure(figsize = (30, 6))
plt.plot(xaxis, mean_ssim_tr, 'tab:blue')
plt.fill_between(xaxis, mean_ssim_tr-std_ssim_tr, mean_ssim_tr+std_ssim_tr,
                 alpha=0.2, facecolor='tab:blue',linewidth=4)
plt.xlabel("Module number (future states)",fontsize=20)
plt.ylabel("SSIM",fontsize=20)
plt.title("Prediction: SSIM on the training set",fontsize=20)
plt.xticks(xaxis,fontsize=20)
plt.savefig(path + 'ssimpred_train.png', dpi=300)
plt.yticks(fontsize=20)

#%% plotting both training and test set together
xaxis = np.arange(5,nfuture+5)
xaxis_ticks = np.array([5,10,15,20,25,30,35,40,45,48])
plt.figure(figsize = (30, 6))
plt.plot(xaxis, mean_mse_tr, 'tab:blue')
plt.plot(xaxis, mean_mse, 'tab:orange')

plt.fill_between(xaxis, mean_mse_tr-std_mse_tr, mean_mse_tr+std_mse_tr,
                 alpha=0.2, facecolor='tab:blue',linewidth=4)
plt.fill_between(xaxis, mean_mse-std_mse, mean_mse+std_mse,
                 alpha=0.2, facecolor='tab:orange',linewidth=4)

plt.xlabel("Module number (future states)",fontsize=28)
plt.ylabel("MSE",fontsize=28)
plt.title("Forecasting: MSE on the training and test set",fontsize=28)
plt.xticks(xaxis_ticks,fontsize=28)
plt.yticks(fontsize=28)
plt.legend(['Train','Test'],fontsize=24,loc='upper left')
plt.xlim([5,48])
plt.savefig(path + 'msepred_traintest.png', bbox_inches='tight', dpi=300)
plt.show()


plt.figure(figsize = (30, 6))
plt.plot(xaxis, mean_ssim_tr, 'tab:blue')
plt.plot(xaxis, mean_ssim, 'tab:orange')

plt.fill_between(xaxis, mean_ssim_tr-std_ssim_tr, mean_ssim_tr+std_ssim_tr,
                 alpha=0.2, facecolor='tab:blue',linewidth=4)
plt.fill_between(xaxis, mean_ssim-std_ssim, mean_ssim+std_ssim,
                 alpha=0.2, facecolor='tab:orange',linewidth=4)

plt.xlabel("Module number (future states)",fontsize=28)
plt.ylabel("SSIM",fontsize=28)
plt.title("Forecasting: SSIM on the training and test set",fontsize=28)
plt.xticks(xaxis_ticks,fontsize=28)
plt.yticks(fontsize=28)
plt.legend(['Train','Test'],fontsize=24,loc='upper left')
plt.xlim([5,48])
plt.savefig(path + 'ssimpred_traintest.png', bbox_inches='tight', dpi=300)
plt.show()

#%% =====>>>> LATENT SPACE TRAJECTORIES <<<<======
############################################################################################
#########################  TRAJECTORIES PART ###############################################
############################################################################################

#%% latent space trajectories - PCA
# obtain the past and future ls samples
ls_all_app = []
for i in range(test_lc_rs.shape[0]):
    xpast_ls = torch.from_numpy(test_lc_rs[i,past1:past2,:])
    xpast_ls = xpast_ls.reshape((1, past2-past1, nfeatures)).to(device)
    xpast_ls = xpast_ls.clone().to(torch.float32)
    length_ls = torch.tensor([xpast_ls.shape[1]])
    
    nfuture = 48-(past1+past2)
    future_ls_all, decoded_all = RECURSION(rnn, cvae15, xpast_ls, length_ls, device, past1, past2, nfuture)
    gen_all = np.array(decoded_all)
    
    # combine past values (inputs) & predicted future values
    xpast_ls = torch.squeeze(xpast_ls,0).detach().cpu().numpy()
    future_ls_all = np.squeeze(future_ls_all, 1)

    ls_all = np.concatenate((xpast_ls, future_ls_all),axis=0)
    ls_all_app.append(ls_all)
    
ls_all_app = np.array(ls_all_app)

# reshape in 2d array
ls_all_rs = ls_all_app.reshape(100*48,8)
test_lc = test_lc_rs.reshape(100*48,8)

# use the same projection matrix to see PCA for forecasted samples
latent_forecast_centered = ls_all_rs - mean_X
latent_forecast_projected = latent_forecast_centered.dot(projection_matrix)
latent_forecast_pca = create_reflection_matrix(latent_forecast_projected)

# use the same projection matrix to see PCA for original samples
latent_test_centered = test_lc - mean_X
latent_test_projected = latent_test_centered.dot(projection_matrix)
latent_test_pca = create_reflection_matrix(latent_test_projected)

# reshape into a 3d array
latent_forecast_pca_rs = latent_forecast_pca.reshape(100,48,2)
latent_test_pca_rs = latent_test_pca.reshape(100,48,2)

# trajactors in LS
from lib.trajectors_forecasting import TRAJECTORIESinLATENT
modidx, prt = 10, 0
TRAJECTORIESinLATENT(Xc_pca, latent_test_pca_rs, latent_forecast_pca_rs, modidx, prt, title, 'pca')
plt.savefig(path + 'latentsp_pca_trajectories.png', bbox_inches='tight', dpi=300)

#%% latent space trajectories - T-SNE
combine_data_ls = np.concatenate((latent_combine, ls_all_rs, test_lc))
size_train = latent_combine.shape[0]
size_test = test_lc.shape[0]

tsne = TSNE(n_components=2, learning_rate='auto',
            init ='pca').fit_transform(combine_data_ls)
Xc_tsne = tsne[0:size_train,:]
Xt_tsne = tsne[size_train:size_train+size_test,:]
Xls_tsne = tsne[size_train+size_test:,:]

Xt_tsne = Xt_tsne.reshape(100,48,2)
Xls_tsne = Xls_tsne.reshape(100,48,2)

modidx, prt = 10, 0
TRAJECTORIESinLATENT(Xc_tsne, Xt_tsne, Xls_tsne, modidx, prt, title, 'tsne')
plt.savefig(path + 'latentsp_tsne_trajectories.png', bbox_inches='tight', dpi=300)

path = 'E:/MSR/codes/RFLA_VAE_LSTM_pyt/code_sent_to_alex/2024_02_26/data/'
np.save(path + 'latentsp_tsne_trainvaldata', Xc_tsne)
np.save(path + 'latentsp_tsne_testdata', Xt_tsne)
np.save(path + 'latentsp_tsne_forecasteddata', Xls_tsne)

#%% parallel coords trajectories
path = 'E:/MSR/codes/RFLA_VAE_LSTM_pyt/code_sent_to_alex/2024_02_26/data/'
N1 = np.ones(latent_test.shape[0],dtype='int64')
N2 = np.ones(ls_all_rs.shape[0],dtype='int64')*2
coloring = np.concatenate((N1, N2),axis=0)
Nc = np.concatenate((latent_per_mod, ls_gensamples_plt),axis=0)
parallelcords2(Nc, coloring, bgcolor, all_colors[1], n_mod)
#plt.savefig(path+'latent_parallelcords_gen_'+'mod_'+str(n_mod+1)+'.png', dpi=300)