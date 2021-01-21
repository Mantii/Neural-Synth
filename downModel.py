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
        self.meanL=nn.Sequential(
            nn.Linear(256,256),nn.BatchNorm1d(256),nn.ReLU(),
            nn.Linear(256,256),nn.BatchNorm1d(256),nn.ReLU(),
            nn.Linear(256,128),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Linear(128,128),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Linear(128,128),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Linear(128,127),nn.BatchNorm1d(127),nn.ReLU()
        )
        self.sigmaL=nn.Sequential(
            nn.Linear(256,256),nn.BatchNorm1d(256),nn.ReLU(),
            nn.Linear(256,256),nn.BatchNorm1d(256),nn.ReLU(),
            nn.Linear(256,128),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Linear(128,128),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Linear(128,128),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Linear(128,127),nn.BatchNorm1d(127),nn.ReLU()
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
        x=x[0:x.shape[0],0,0:1024].view(x.shape[0],1,1024)

        for conv in self.encoder:
            x=conv(x)

        x=x.view(x.shape[0],256)
        z=self.sample_latent(x,cl)

        
        
        x=self.LinDecoder(z)
        x=x.view(x.shape[0],1,512)
        return x