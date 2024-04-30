import numpy as np
import torch

def RECURSION(rnn, cvae, past_ls, length_ls, device, past1, past2, nfuture):
    future_all = []
    decoded_all = []
    rnn.eval()
    cvae = cvae.to(device)
    cvae.eval()
    for i in range(nfuture):
        with torch.no_grad():
            future_ls = rnn(past_ls,length_ls,device)
            decoded_img = cvae.decode(future_ls)
            decoded_img = np.squeeze(decoded_img,0)
            decoded_all.append(decoded_img.cpu().detach())
            past_ls = np.squeeze(past_ls,0)
            past_ls = torch.cat((past_ls,future_ls),0)
            past_ls = past_ls[np.newaxis,:,:]
            length_ls = torch.tensor([past_ls.shape[1]])
            future_all.append(future_ls.detach().cpu().numpy())   
    return np.array(future_all), decoded_all