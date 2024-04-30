import numpy as np

def pad_dataset(dataset,steps,maxlen,padvalue):
    nwindows = steps.shape[0]
    padded_dataset = []
    all_len = []
    for i in range(nwindows):
        # arr = (number of windows,window size,latent dim)
        arr = dataset[i]
        shp = steps[i]
        ones_arr = np.ones(arr.shape[0],dtype='int64')*shp
        arr = np.moveaxis(arr,1,2)
        padded = np.ones((arr.shape[0],arr.shape[1],maxlen))*padvalue
        for j in range(arr.shape[0]):
            sample = arr[j,:,:]
            padded[j,0:sample.shape[0],0:sample.shape[1]] = sample
            
        padded = np.moveaxis(padded,1,2)
        padded_dataset.append(padded)
        all_len.append(ones_arr)
        
    return np.concatenate(padded_dataset,axis=0), np.concatenate(all_len,dtype='int64')