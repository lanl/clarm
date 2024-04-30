import numpy as np

def PCAbuild(Xtrain, n):
    # Assume 'X' is your original data matrix with samples in rows and features in columns
    
    # Step 1: Center the data
    mean_X = np.mean(Xtrain, axis=0)
    centered_X = Xtrain - mean_X
    
    # Step 2: Compute the covariance matrix
    covariance_matrix = np.cov(centered_X, rowvar=False)
    
    # Step 3: Compute eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Step 4: Sort eigenvectors by eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors_sorted = eigenvectors[:, sorted_indices]
    
    # Step 5: Select the top k eigenvectors
    projection_matrix = eigenvectors_sorted[:, :2]
    
    return (mean_X, projection_matrix)
    
    # How to use this
    '''centered_X = latent_combine - mean_X
       X_projected = centered_X.dot(projection_matrix)'''
