import numpy as np
import matplotlib.pyplot as plt

def COLORPLOTSinLATENT(latent,perturb_all,title,label):
    mod_number = np.zeros(latent.shape[0])
    count = 0
    for n in np.arange(perturb_all):
        for n2 in np.arange(48):
            mod_number[count] = n2+1
            count = count + 1
    plt.figure(figsize=(6, 6))
    plt.title(title, fontsize=20)
    plt.scatter(latent[:, 0], latent[:, 1], s=15, c=mod_number, cmap='jet')
    plt.xlabel(label + str(1), fontsize=20)
    plt.ylabel(label + str(2), fontsize=20)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()