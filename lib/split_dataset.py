# used after SPLIT_SEQUENCE function
import numpy as np
from lib.split_sequence import SPLIT_SEQUENCES

def SPLIT_DATASET(dataset,steps,length):
    Xdataset = []
    Ydataset = []
    for i in steps:
        # for every step (window)
        xdata = []
        ydata = []
        for j in range(length):
            # for every perturbation
            x,y = SPLIT_SEQUENCES(dataset[j,:,:], i)
            xdata.append(x)
            ydata.append(y)   
        xdata = np.concatenate(xdata,axis=0)
        ydata = np.concatenate(ydata,axis=0)
        Xdataset.append(xdata)
        Ydataset.append(ydata)
    return (Xdataset,Ydataset)