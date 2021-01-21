import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable

class AutoEncoder(nn.Module):
    def __init__(self,):
        super(AutoEncoder,self).__init__()
        
        self.normal=torch.distributions.normal.Normal(1,1)
        
        self.encoder=nn.ModuleList([
            nn.Conv1d(1,1,16,8,padding=1,dilation=2),
            nn.Sigmoid(),
            nn.Conv1d(1,128,16,8,padding=1,dilation=2),
            nn.Sigmoid(),
            nn.Conv1d(128,128,16,8,padding=1,dilation=2),
            nn.Sigmoid(),
            nn.Conv1d(128,1024,16,8,padding=1,dilation=2),
            nn.Sigmoid(),
            nn.MaxPool1d(12)
            
            
        ])
        self.meanL=nn.Sequential(
            nn.Linear(1024,1023)
        )
        self.sigmaL=nn.Sequential(
            nn.Linear(1024,1023)
            )
        self.DecoderFC=nn.Sequential(
            nn.Linear(1024,1024),nn.Sigmoid(),
            nn.Linear(1024,1024),nn.Sigmoid(),
            nn.Linear(1024,1024),nn.Sigmoid(),
            nn.Linear(1024,1024),nn.Sigmoid(),
            nn.Linear(1024,1024),nn.Sigmoid()
        )
        self.decoder=nn.ModuleList([
            nn.ConvTranspose1d(1024,1024,8,4,1,dilation=2), nn.Sigmoid(),
            nn.ConvTranspose1d(1024,128,16,8,2,dilation=2), nn.Sigmoid(),
            nn.ConvTranspose1d(128,128,16,8,5,dilation=2), nn.Sigmoid(),
            nn.ConvTranspose1d(128,1,16,8,1,dilation=2), nn.Sigmoid(),
            nn.ConvTranspose1d(1,1,16,8,0,output_padding=1,dilation=2), nn.Sigmoid()
        ])

        self.weight1 = Parameter(self.lastWeightTensor())
    def lastWeightTensor(self):
        batch_size=16
        x=torch.zeros(batch_size*64000).view(batch_size,64000)
        x[:][::2]=0.1
        x[:][1::2]=-0.2
        return x
    def sample_latent(self,x,cl):
        mean=self.meanL(x)
        sigma=self.sigmaL(x)
        
        self.mean=mean
        self.sigma=sigma
        eps = torch.distributions.normal.Normal(0, 1).sample(sample_shape=sigma.size())
        z=mean+sigma*Variable(eps,requires_grad=False).cuda()
        z=torch.cat((z,cl),dim=1)
        return z
    
    def forward(self,x):
        cl=x[0:x.shape[0],0,-1].view(x.shape[0],1)
        x=x[0:x.shape[0],0,0:64000].view(x.shape[0],1,64000)
        
        for conv in self.encoder:
            x=conv(x)
            #print(x.shape)
        x=x.view(x.shape[0],1024)
        
        z=self.sample_latent(x,cl)
        z=self.DecoderFC(z)
        x=z.view(z.shape[0],1024,1)
        
        
        for conv in self.decoder:
            x=conv(x)
            #print(x.shape)
        #x=torch.add(x.view(x.shape[0],64000),self.weight1)

        return x