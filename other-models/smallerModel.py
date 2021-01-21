import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        

        
        self.encoder=nn.ModuleList([
            nn.Conv1d(1,128,16,4,padding=0,dilation=2),
            nn.ReLU(),
            nn.Conv1d(128,256,16,4,padding=0,dilation=2),
            nn.ReLU(),
            nn.Conv1d(256,512,16,4,padding=0,dilation=2),
            nn.ReLU(),
            nn.Conv1d(512,1024,16,4,padding=0,dilation=2),
            nn.ReLU(),
            nn.Conv1d(1024,1024,8,4,padding=0,dilation=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
            
            
        ])
        
        
        self.meanL=nn.Sequential(
            nn.Linear(1024,1024),nn.ReLU(),
            nn.Linear(1024,512),nn.ReLU(),
            nn.Linear(512,512),nn.ReLU(),
            nn.Linear(512,511)

    

        )
        self.sigmaL=nn.Sequential(
            nn.Linear(1024,1024),nn.ReLU(),
            nn.Linear(1024,512),nn.ReLU(),
            nn.Linear(512,512),nn.ReLU(),
            nn.Linear(512,511)

        )
        self.DecoderFC=nn.Sequential(
            nn.Linear(512,512),nn.ReLU(),
            nn.Linear(512,1024),nn.ReLU(),
            nn.Linear(1024,1024),nn.ReLU()

        )
        self.decoder=nn.ModuleList([
            nn.ConvTranspose1d(1024,1024,3,1,0,dilation=1), nn.ReLU(),
            nn.ConvTranspose1d(1024,1024,8,4,0,dilation=2), nn.ReLU(),
            nn.ConvTranspose1d(1024,512, 16,4,1,dilation=2), nn.ReLU(),
            nn.ConvTranspose1d(512, 256, 16,4,2,dilation=2), nn.ReLU(),
            nn.ConvTranspose1d(256, 128, 16,4,0,output_padding=2,dilation=2), nn.ReLU(),
            nn.ConvTranspose1d(128, 1,   16,4,0,output_padding=1,dilation=2)

        ])
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
        x=x[0:x.shape[0],0,0:8000].view(x.shape[0],1,8000)
        
        for conv in self.encoder:
            x=conv(x)
            print(x.shape)
        x=x.view(x.shape[0],1024)
        
        z=self.sample_latent(x,cl)
        z=self.DecoderFC(z)
        x=z.view(z.shape[0],1024,1)
        
        
        for conv in self.decoder:
            x=conv(x)
            print(x.shape)
        #x=torch.add(x.view(x.shape[0],64000),self.weight1)

        return x
