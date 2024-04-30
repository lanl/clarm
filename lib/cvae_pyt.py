import torch
import torch.nn as nn

class cVAE(nn.Module):

    def __init__(self,imch,f1,f2,f3,f4,f5,
                 n1,n2,n3,
                 d1,imfinal1,imfinal2,device):
        super(cVAE, self).__init__()
        self.device = device

        # Encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(imch, f1, 3, stride=2, padding=1), # imsize = imsize/stride
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f2, 3, stride=2, padding=1),   # imsize = imsize/stride
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(f2),
            nn.Conv2d(f2, f3, 3, stride=2, padding=1),   # imsize = imsize/stride
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(f3),
            nn.Conv2d(f3, f4, 3, stride=2, padding=1),   # imsize = imsize/stride
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(f4),
            nn.Conv2d(f4, f5, 3, stride=2, padding=1),   # imsize = imsize/stride
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(f5))
            # nn.Conv2d(f5, f6, 3, stride=2, padding=1),   # imsize = imsize/stride
            # nn.LeakyReLU(0.2),
            # nn.BatchNorm2d(f6))    
        self.flatten = nn.Flatten(start_dim=1)
        
        # linear layers after CNN
        self.encoder_lin = nn.Sequential(
            nn.Linear(imfinal1 * imfinal2* f5, n1),
            nn.Dropout(0.1), 
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(n1))
        
        # branch with a conditional (module number)
        self.conditional = nn.Sequential(
            nn.Linear(1, n2),nn.LeakyReLU(0.2),
            nn.Linear(n2, n2),nn.LeakyReLU(0.2),
            nn.Linear(n2, n2),nn.LeakyReLU(0.2),
            nn.BatchNorm1d(n2))
        
        # combined linear layers after joining encoder linear and conditinal
        self.combined_lin = nn.Sequential(
            nn.Linear(n1+n2, n3),nn.LeakyReLU(0.2),
            nn.BatchNorm1d(n3))
        
        # latent mean and variance 
        self.mean_layer = nn.Linear(n3, d1)
        self.logvar_layer = nn.Linear(n3, d1)
        
        # Decoder
        self.decoder_lin = nn.Sequential(
            nn.Linear(d1, n1),
            nn.Linear(n1, imfinal1 * imfinal2 *f5),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(imfinal1 * imfinal2 *f5))

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(f5, imfinal1, imfinal2))

        self.decoder_conv = nn.Sequential(
            # nn.ConvTranspose2d(f6, f5, 3, 
            # stride=2, padding = 1, output_padding=1),
            # nn.LeakyReLU(0.2),
            # nn.BatchNorm2d(f5),
            nn.ConvTranspose2d(f5, f4, 3, 
            stride=2, padding = 1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(f4),
            nn.ConvTranspose2d(f4, f3, 3, stride=2, 
            padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(f3),
            nn.ConvTranspose2d(f3, f2, 3, stride=2, 
            padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(f2),
            nn.ConvTranspose2d(f2, f1, 3, stride=2, 
            padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(f1),
            nn.ConvTranspose2d(f1, imch, 3, stride=2, 
            padding=1, output_padding=1))
    
    def encode(self, x, c):
        x = self.encoder_cnn(x)
        #print('encoder cnn',x.shape)
        x = self.flatten(x)
        #print('flatten',x.shape)
        x = self.encoder_lin(x)
        #print('encoder_lin',x.shape)
        xb = self.conditional(c)
        xc = torch.cat((x, xb),axis=1)
        #print("combined",xc.shape)
        x = self.combined_lin(xc)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        if self.training:
            epsilon = torch.randn_like(var)      
            z = mean + var*epsilon
            return z
        else:
            return mean

    def decode(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

    def forward(self, x, c):
        mean, logvar = self.encode(x,c)
        z = self.reparameterization(mean, torch.exp(0.5 * logvar))
        x_hat = self.decode(z)
        return x_hat, mean, logvar, z