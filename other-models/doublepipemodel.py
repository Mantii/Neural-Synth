import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        
        
        self.encoder=nn.ModuleList([
            nn.Conv1d(1,8,8,4,0,dilation=2),nn.BatchNorm1d(8), nn.ReLU(),
            nn.Conv1d(8,64,8,4,0,dilation=2),nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64,128,8,4,0,dilation=2),nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128,256,4,2,0,dilation=1),nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256,256,4,2,0,dilation=1), nn.BatchNorm1d(256),nn.ReLU(),  
        ])
        self.meanL1=nn.Sequential(
            nn.Linear(256,256),nn.BatchNorm1d(256),nn.ReLU(),
            nn.Linear(256,128),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Linear(128,128),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Linear(128,64),nn.BatchNorm1d(64),nn.ReLU(),
            nn.Linear(64,64),nn.BatchNorm1d(64),nn.ReLU(),
            nn.Linear(64,32),nn.BatchNorm1d(32),nn.ReLU(),
            nn.Linear(32,32),nn.BatchNorm1d(32),nn.ReLU()
        )
        self.sigmaL1=nn.Sequential(
            nn.Linear(256,256),nn.BatchNorm1d(256),nn.ReLU(),
            nn.Linear(256,128),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Linear(128,128),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Linear(128,64),nn.BatchNorm1d(64),nn.ReLU(),
            nn.Linear(64,64),nn.BatchNorm1d(64),nn.ReLU(),
            nn.Linear(64,32),nn.BatchNorm1d(32),nn.ReLU(),
            nn.Linear(32,32),nn.BatchNorm1d(32),nn.ReLU()
        )
        self.meanL2=nn.Sequential(
            nn.Linear(256,256),nn.BatchNorm1d(256),nn.ReLU(),
            nn.Linear(256,128),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Linear(128,128),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Linear(128,64),nn.BatchNorm1d(64),nn.ReLU(),
            nn.Linear(64,64),nn.BatchNorm1d(64),nn.ReLU(),
            nn.Linear(64,32),nn.BatchNorm1d(32),nn.ReLU(),
            nn.Linear(32,32),nn.BatchNorm1d(32),nn.ReLU()
        )
        self.sigmaL2=nn.Sequential(
            nn.Linear(256,256),nn.BatchNorm1d(256),nn.ReLU(),
            nn.Linear(256,128),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Linear(128,128),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Linear(128,64),nn.BatchNorm1d(64),nn.ReLU(),
            nn.Linear(64,64),nn.BatchNorm1d(64),nn.ReLU(),
            nn.Linear(64,32),nn.BatchNorm1d(32),nn.ReLU(),
            nn.Linear(32,32),nn.BatchNorm1d(32),nn.ReLU()
        )        
        self.dec1=nn.Sequential(
            nn.Linear(32,32),nn.BatchNorm1d(32),nn.ReLU(),
            nn.Linear(32,64),nn.BatchNorm1d(64),nn.ReLU(),
            nn.Linear(64,64),nn.BatchNorm1d(64),nn.ReLU(),
            nn.Linear(64,64),nn.BatchNorm1d(64),nn.ReLU()
  
        )     
        self.dec2=nn.Sequential(
            nn.Linear(32,32),nn.BatchNorm1d(32),nn.ReLU(),
            nn.Linear(32,64),nn.BatchNorm1d(64),nn.ReLU(),
            nn.Linear(64,64),nn.BatchNorm1d(64),nn.ReLU(),
            nn.Linear(64,64),nn.BatchNorm1d(64),nn.ReLU()
  
        )     
        self.LinDecoder=nn.Sequential(
            nn.Linear(128,128),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Linear(128,128),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Linear(128,256),nn.BatchNorm1d(256),nn.ReLU(),
            nn.Linear(256,256),nn.BatchNorm1d(256),nn.ReLU(),
            nn.Linear(256,256),nn.BatchNorm1d(256),nn.ReLU(),
            nn.Linear(256,512),nn.BatchNorm1d(512),nn.ReLU(),
            nn.Linear(512,512),nn.BatchNorm1d(512),nn.ReLU(),
            nn.Linear(512,512)
        )
        self.up=nn.Upsample(scale_factor=2)
        self.UpDec=nn.Sequential(
            nn.Linear(1024,1024),nn.ReLU(),
            nn.Linear(1024,1024)
        )
        

    def sample_latent(self,x):
        mean1=self.meanL1(x)
        sigma1=torch.sqrt(torch.exp(self.sigmaL1(x)))
        mean2=self.meanL2(x)
        sigma2=torch.sqrt(torch.exp(self.sigmaL2(x)))
        self.mean=(mean1+mean2)/2
        self.sigma=(sigma1+sigma2)/2
        eps = torch.distributions.normal.Normal(0, 1).sample(sample_shape=sigma1.size())
        z1=mean1+sigma1*Variable(eps,requires_grad=False).cuda()
        z2=mean1+sigma1*Variable(eps,requires_grad=False).cuda()

        z=torch.cat((self.dec1(z1),self.dec2(z2)),dim=1)
        return z
    
    def forward(self,x):
        x=x.view(x.shape[0],1,1024)

        for conv in self.encoder:
            x=conv(x)

        x=x.view(x.shape[0],256)
        z=self.sample_latent(x)

        
        
        x=self.LinDecoder(z)
        x=x.view(x.shape[0],1,512)
        x=self.up(x)
        x=self.UpDec(x)
        return x