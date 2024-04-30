import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np

def PC_modwise(data, nlines_permod):
    fig, host = plt.subplots(figsize=(15,7))
    
    # create some dummy data
    ynames = ['$Z_1$', '$Z_2$', '$Z_3$', '$Z_4$', '$Z_5$', '$Z_6$', '$Z_7$', '$Z_8$']
    length = data.shape[0]
    
    y1 = data[:,0]
    y2 = data[:,1]
    y3 = data[:,2]
    y4 = data[:,3]
    y5 = data[:,4]
    y6 = data[:,5]
    y7 = data[:,6]
    y8 = data[:,7]
    
    # organize the data
    ys = np.dstack([y1, y2, y3, y4, y5, y6, y7, y8])[0]
    ymins = ys.min(axis=0)
    ymaxs = ys.max(axis=0)
    ymins_all = ymins.min()
    ymaxs_all = ymaxs.max()
    print('ymin', ymins_all)
    print('ymax', ymaxs_all)
    dys = ymaxs - ymins
    ymins -= dys * 0.05  # add 5% padding below and above
    ymaxs += dys * 0.05
    dys = ymaxs - ymins
    
    # transform all data to be compatible with the main axis
    zs = np.zeros_like(ys)
    zs[:, 0] = ys[:, 0]
    zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]
    
    axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]
    for i, ax in enumerate(axes):
        ax.set_ylim(ymins[i], ymaxs[i])
        #ax.set_ylim(ymins_all, ymaxs_all)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        #ax.set_yticks([-2,-1,0,1,2])
        ax.tick_params(axis = 'y', labelsize=18)
        if ax != host:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks_position('right')
            ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))
    
    host.set_xlim(0, ys.shape[1] - 1)
    host.set_xticks(range(ys.shape[1]))
    host.set_xticklabels(ynames, fontsize=18)
    host.tick_params(axis='x', which='major', pad=7)
    host.spines['right'].set_visible(False)
    host.xaxis.tick_top()
      
    cmap = plt.cm.jet
    colors = cmap(np.linspace(0, 1, 48))
    colors_3ch = colors[:,0:3]
    colors_rep = np.repeat(colors_3ch, nlines_permod, axis=0)
    
    for j in range(length):
        verts = list(zip([x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
                         np.repeat(zs[j, :], 3)[1:-1]))
        # for x,y in verts: host.plot(x, y, 'go') # to show the control points of the beziers
        codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
        path = Path(verts, codes)
        
        color = colors_rep[j]
        width = 0.5
        alp = 1

        patch = patches.PathPatch(path, facecolor='none', lw=width, alpha=alp, edgecolor=color)
        host.add_patch(patch)
    
    #plt.colorbar(cmap)
    plt.tight_layout()