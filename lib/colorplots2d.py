import numpy as np
import matplotlib.pyplot as plt

def COLORPLOTS_2D(latent,perturb_all,dim1,dim2,title):
    mod_number = np.zeros(latent.shape[0])
    count = 0
    for n in np.arange(perturb_all):
        for n2 in np.arange(48):
            mod_number[count] = n2+1
            count = count + 1
    plt.figure(figsize=(6, 6))
    plt.title(title, fontsize=20)
    plt.scatter(latent[:, dim1], latent[:, dim2], s=15, c=mod_number, cmap='jet')
    plt.xlabel('$Z_' + str(dim1+1) + '$', fontsize=20)
    plt.ylabel('$Z_' + str(dim2+1) + '$', fontsize=20)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()