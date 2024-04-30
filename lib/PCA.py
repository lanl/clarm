from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA  
import numpy as np

class PCAmodel:
    @staticmethod
    def pcabuild(data,dim):
        nd = dim
        pca_l = PCA(n_components=nd, svd_solver= 'full', whiten=True)
        pca_l.fit(data)
        trans_pca_l = pca_l.transform(data)
        
        explained_var = np.var(trans_pca_l, axis=0)
        explained_var_ratio = explained_var / np.sum(explained_var)
        data_reduced = pca_l.inverse_transform(trans_pca_l)

        return (trans_pca_l, explained_var_ratio, data_reduced)
    
    def build_nonlin(data,dim):
        nd = dim
        pca_nl = KernelPCA(n_components = nd, kernel= 'linear', 
                           fit_inverse_transform=True)
        pca_nl.fit(data)
        trans_pca_nl = pca_nl.transform(data)
        
        explained_var = np.var(trans_pca_nl, axis=0)
        explained_var_ratio = explained_var / np.sum(explained_var)
        data_reduced = pca_nl.inverse_transform(trans_pca_nl)
        
        return (trans_pca_nl, explained_var_ratio, data_reduced)