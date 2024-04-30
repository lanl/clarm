import numpy as np
import matplotlib.pyplot as plt

def TRAJECTORIESinLATENT(latent, modidx, prt, title, label):
    mod_number = np.zeros(latent.shape[0])
    count = 0
    nxtpert = 48
    for n in np.arange(int(latent.shape[0]/48)):
        for n2 in np.arange(48):
            mod_number[count] = n2+1
            count = count + 1
            
    plt.figure(figsize=(6, 6))
    plt.scatter(latent[:, 0], latent[:, 1],  s=15, c=mod_number, cmap='jet')  
    plt.colorbar()
      
    for prtbidx in np.arange(0,prt):
        #plt.figure(figsize=(6, 6))
        #plt.scatter(latent[0:modidx, 0], latent[0:modidx, 1], marker = '.')
        #plt.plot(latent[0:modidx, 0], latent[0:modidx, 1],'--r',linewidth=3, markersize=12)
        
        for i in np.arange(0, modidx):
            plt.text(latent[i, 0], latent[i, 1], str(i), fontsize=30, c = 'r')

        plt.scatter(latent[0+prtbidx*nxtpert:modidx+prtbidx*nxtpert, 0],
                    latent[0+prtbidx*nxtpert:modidx+prtbidx*nxtpert, 1], 
                    marker = '.')
        plt.plot(latent[0+prtbidx*nxtpert:modidx+prtbidx*nxtpert, 0],
                  latent[0+prtbidx*nxtpert:modidx+prtbidx*nxtpert, 1],'--',c='tab:red',
                  linewidth=3, markersize=12)
        
        #for i in np.arange(0+prtbidx*nxtpert,modidx+prtbidx*nxtpert):
            #plt.text(latent[i, 0], latent[i, 1], str(i-prtbidx*nxtpert),
                     #fontsize=30)
            
        plt.title(title, fontsize = 20)
        plt.xlabel(label + str(1), fontsize=20)
        plt.ylabel(label + str(2), fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()    