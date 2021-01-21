import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        
        self.first=nn.Sequential(
            nn.Linear(1024,1024),nn.ReLU())
        
        
        self.meanL=nn.Sequential(
            nn.Linear(1024,1024),nn.ReLU(),
            nn.Linear(1024,512),nn.ReLU(),
            nn.Linear(512,512),nn.ReLU(),
            nn.Linear(512,256),nn.ReLU(),
            nn.Linear(256,256),nn.ReLU(),
            nn.Linear(256,128),nn.ReLU(),
            nn.Linear(128,128),nn.ReLU(),
            nn.Linear(128,127)


        )
        self.sigmaL=nn.Sequential(
            nn.Linear(1024,1024),nn.ReLU(),
            nn.Linear(1024,512),nn.ReLU(),
            nn.Linear(512,512),nn.ReLU(),
            nn.Linear(512,256),nn.ReLU(),
            nn.Linear(256,256),nn.ReLU(),
            nn.Linear(256,128),nn.ReLU(),
            nn.Linear(128,128),nn.ReLU(),
            nn.Linear(128,127)


        )
        self.DecoderFC=nn.Sequential(
            nn.Linear(128,128),nn.ReLU(),
            nn.Linear(128,128),nn.ReLU(),
            nn.Linear(128,128),nn.ReLU(),
            nn.Linear(128,128),nn.ReLU(),
            nn.Linear(128,128),nn.ReLU()


        )
        self.decoder=nn.ModuleList([
            nn.ConvTranspose1d(128,64,  8, 2,0,dilation=2), nn.ReLU(),
            nn.ConvTranspose1d(64, 32,  8, 2,0,dilation=2), nn.ReLU(),
            nn.ConvTranspose1d(32, 16,  8, 2,0,output_padding=0,dilation=2), nn.ReLU(),
            nn.ConvTranspose1d(16, 8,   8, 2,0,dilation=2), nn.ReLU(),
            nn.ConvTranspose1d(8,  4,   8, 2,0,output_padding=0,dilation=2), nn.ReLU(),
            nn.ConvTranspose1d(4,  2,   8, 2,0,output_padding=0,dilation=2), nn.ReLU(),
            nn.ConvTranspose1d(2,  1,   8, 2,0,output_padding=0,dilation=2), nn.ReLU(),

        ])
        self.fc1=nn.Sequential(
            nn.Linear(1779,1024),nn.ReLU(),
            nn.Linear(1024,1024),nn.ReLU(),
            nn.Linear(1024,1024)

            )
    def sample_latent(self,x,cl):
        mean=self.meanL(x)
        sigma=self.sigmaL(x)
        sigma=torch.sqrt(torch.exp(sigma))
        self.mean=mean
        self.sigma=sigma
        eps = torch.distributions.normal.Normal(0, 1).sample(sample_shape=sigma.size())
        z=mean+sigma*Variable(eps,requires_grad=False).cuda()
        z=torch.cat((z,cl),dim=1)
        return z
    
    def forward(self,x):
        cl=x[0:x.shape[0],0,-1].view(x.shape[0],1)
        x=x[0:x.shape[0],0,0:1024].view(x.shape[0],1024)
        x=self.first(x)
        
        z=self.sample_latent(x,cl)
        x=z.view(z.shape[0],128,1)
        
        
        for conv in self.decoder:
            x=conv(x)

        x=self.fc1(x.view(x.shape[0],1779))
        return x.view(x.shape[0],1,1024)
