import torch
from scipy.linalg import sqrtm
import numpy as np

def calculate_activation_statistics(images):
    # Calculate ResNet activations and statistics from the specified layer
    mean = torch.mean(images, dim=0)
    cov = torch_cov(images)
    return mean.numpy(), cov.numpy()

def torch_cov(m, rowvar=False):
    # Calculate covariance matrix in PyTorch
    if m.dim() > 2:
        raise ValueError("m has more than 2 dimensions")
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()
    return fact * m.matmul(mt).squeeze()

def calculate_fid(real_images, gen_images):
    # find mu and sigma
    mu1, sigma1 = calculate_activation_statistics(real_images)
    mu2, sigma2 = calculate_activation_statistics(gen_images)
    
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    return np.trace(sigma1 + sigma2 - 2 * covmean) + np.dot(diff, diff)
