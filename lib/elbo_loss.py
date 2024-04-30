import torch
import torch.nn as nn

def ELBO(x, x_hat, mean, log_var, beta):
    reconstruction_loss_bce = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    kl_loss = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    total_loss = reconstruction_loss_bce + beta*kl_loss
    
    return (total_loss, reconstruction_loss_bce, beta*kl_loss)