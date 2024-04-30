import numpy as np
import cv2

# modify the size of the image
def MODIFYIMSIZE(data,imsize):
    run2ps_resize = np.zeros((data.shape[0],data.shape[1],data.shape[2],data.shape[3]),dtype='float32')
    for i in range(data.shape[0]):
        for j in range(data.shape[3]):
            image = cv2.resize(data[i,:,:,j],(imsize,imsize))
            image = np.float32(image)
            run2ps_resize[i,:,:,j] = image       
    
    return run2ps_resize