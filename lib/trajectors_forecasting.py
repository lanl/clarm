import numpy as np
import matplotlib.pyplot as plt

def TRAJECTORIESinLATENT(latent, org, forecast, modidx, prt, title, label):
    mod_number = np.zeros(latent.shape[0])
    count = 0
    for n in np.arange(int(latent.shape[0]/48)):
        for n2 in np.arange(48):
            mod_number[count] = n2+1
            count = count + 1
            
    plt.figure(figsize=(6, 6))
    plt.scatter(latent[:, 0], latent[:, 1],  s=15, c=mod_number, cmap='jet')  
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)
    
    # forecasting trajectories
    plt.scatter(org[prt, 0:modidx,0], org[prt, 0:modidx, 1], marker = '^', s = 50, c = 'y')
    plt.scatter(forecast[prt, 0:modidx,0], forecast[prt, 0:modidx, 1], marker = '^', s = 50, c = 'r')
    plt.plot(org[prt, 0:modidx, 0], org[prt, 0:modidx, 1],'--',c='y', linewidth=3)
    plt.plot(forecast[prt, 0:modidx, 0], forecast[prt, 0:modidx, 1],'-',c='r', linewidth=3)
    for i in np.arange(0, modidx):
        plt.text(org[prt, i, 0], org[prt, i, 1], str(i+1), fontsize=30, c = 'y')
        plt.text(forecast[prt, i, 0], forecast[prt, i, 1], str(i+1), fontsize=30, c = 'r')
    
    plt.xlabel(label + str(1), fontsize=20)
    plt.ylabel(label + str(2), fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)