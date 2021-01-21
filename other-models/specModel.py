
#source jupenv/pyenv/bin/activate
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        
        self.encoder=nn.ModuleList([
            nn.Conv2d(2,128,(2,4),(1,2)), nn.Sigmoid(),
             nn.Conv2d(128,256,5,5),      nn.Sigmoid(),
             nn.Conv2d(256,512,4,4,1),    nn.Sigmoid(),
             nn.Conv2d(512,1024,2,2,1),   nn.Sigmoid(),
             nn.Conv2d(1024,2048,8,8,1),  nn.Sigmoid()
                    
        ])
        
        self.meanL=nn.Sequential(
            nn.Linear(2048,1024),
            nn.Sigmoid())
        self.sigmaL=nn.Sequential(
            nn.Linear(2048,1024),
            nn.Sigmoid())
        
        self.decoderFC=nn.Sequential(
            nn.Linear(1024,2048),
            nn.Sigmoid() )
        
        self.decoder=nn.ModuleList([

            nn.ConvTranspose2d(2048,1024,8,8,1,1), nn.Sigmoid(),
            nn.ConvTranspose2d(1024,512,2,2,1,1),  nn.Sigmoid(),
            nn.ConvTranspose2d(512,256,4,4,1,1),   nn.Sigmoid(),
            nn.ConvTranspose2d(256,128,5,5),       nn.Sigmoid(),
            nn.ConvTranspose2d(128,1,(2,4),(1,2)), nn.Sigmoid()

        ])
        

                                

    def sample_latent(self,h):
        mean=self.meanL(h)
        sigma=self.sigmaL(h)
        sigma=torch.exp(sigma)
        
        self.mean=mean
        self.sigma=sigma
    
        eps = torch.distributions.normal.Normal(0, 1).sample(sample_shape=sigma.size())
        z=mean+sigma*Variable(eps,requires_grad=False)
        return z
    
    def forward(self,x):
        
        for conv in self.encoder:
            x=conv(x)

            
        z=self.sample_latent(x.view(-1,2048))
        x=self.decoderFC(z)
        x=x.view(-1,2048,1,1)
        for convt in self.decoder:
            x=convt(x)
        return x
