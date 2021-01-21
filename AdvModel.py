import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        stride=2
        ker=2
        channels=128
        self.convs=nn.ModuleList([
            nn.ReLU(),
            nn.Conv1d(1,channels,1,1,0,dilation=1),        nn.ReLU(),nn.Conv1d(channels,channels,ker,stride,0,dilation=2),nn.ReLU(),
            nn.Conv1d(channels,channels,1,1,0,dilation=1), nn.ReLU(),nn.Conv1d(channels,channels,ker,stride,0,dilation=2),nn.ReLU(),
            nn.Conv1d(channels,channels,1,1,0,dilation=1), nn.ReLU(),nn.Conv1d(channels,channels,ker,stride,0,dilation=2),nn.ReLU(),
            nn.Conv1d(channels,channels,1,1,0,dilation=1), nn.ReLU(),nn.Conv1d(channels,channels,ker,stride,0,dilation=2),nn.ReLU(),
            nn.Conv1d(channels,channels,1,1,0,dilation=1), nn.ReLU(),nn.Conv1d(channels,channels,ker,stride,0,dilation=2),nn.ReLU(),
            nn.Conv1d(channels,channels,1,1,0,dilation=1), nn.ReLU(),nn.Conv1d(channels,channels,ker,stride,0,dilation=2),nn.ReLU(),
            nn.Conv1d(channels,channels,1,1,0,dilation=1), nn.ReLU(),nn.Conv1d(channels,channels,ker,stride,0,dilation=2),nn.ReLU(),
            nn.Conv1d(channels,channels,1,1,0,dilation=1), nn.ReLU(),nn.Conv1d(channels,channels,ker,stride,0,dilation=2),nn.ReLU(),
            nn.Conv1d(channels,channels,1,1,0,dilation=1), nn.ReLU(),nn.Conv1d(channels,channels,ker,stride,0,dilation=2),nn.ReLU(),
            nn.Conv1d(channels,channels,1,1,0,dilation=1), nn.ReLU(),nn.Conv1d(channels,channels,ker,stride,0,dilation=2),nn.ReLU(),
            nn.Conv1d(channels,channels,1,1,0,dilation=1), nn.AvgPool1d(61)
        ])
        self.meanL1=nn.Sequential(
            nn.Linear(128,128),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Linear(128,128),nn.BatchNorm1d(128),nn.ReLU()
        )
        self.sigmaL1=nn.Sequential(
            nn.Linear(128,128),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Linear(128,128),nn.BatchNorm1d(128),nn.ReLU()
        )
        self.meanL2=nn.Sequential(
            nn.Linear(128,128),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Linear(128,128),nn.BatchNorm1d(128),nn.ReLU()
        )
        self.sigmaL2=nn.Sequential(
            nn.Linear(128,128),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Linear(128,128),nn.BatchNorm1d(128),nn.ReLU()
        )
        self.meanL3=nn.Sequential(
            nn.Linear(128,128),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Linear(128,128),nn.BatchNorm1d(128),nn.ReLU()
        )
        self.sigmaL3=nn.Sequential(
            nn.Linear(128,128),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Linear(128,128),nn.BatchNorm1d(128),nn.ReLU()
        )
        self.pitchDec=nn.Sequential(
            nn.Linear(129,129),nn.BatchNorm1d(129),nn.ReLU(),
            nn.Linear(129,129),nn.BatchNorm1d(129),nn.ReLU()
        )
        self.qualDec=nn.Sequential(
            nn.Linear(138,138),nn.BatchNorm1d(138),nn.ReLU(),
            nn.Linear(138,138),nn.BatchNorm1d(138),nn.ReLU()
        )

        self.classDec=nn.Sequential(
            nn.Linear(138,138),nn.BatchNorm1d(138),nn.ReLU(),
            nn.Linear(138,138),nn.BatchNorm1d(138),nn.ReLU()
        )
        self.jointDec=nn.Sequential(
            nn.Linear(405,512),nn.BatchNorm1d(512),nn.ReLU(),
            nn.Linear(512,512),nn.BatchNorm1d(512),nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Linear(512,1000),nn.ReLU(),
            nn.Linear(1000,1000),nn.ReLU(),
            nn.Upsample(scale_factor=2),

        )
        self.upDec=nn.Sequential(
            nn.Conv1d(1,1).nn.ReLU(),
            nn.Upsample(scale_factor=2), #2000
            nn.Conv1d(1,1).nn.ReLU(),
            nn.Upsample(scale_factor=2), #4000
            nn.Conv1d(1,1).nn.ReLU(),
            nn.Upsample(scale_factor=2), #8000
            nn.Conv1d(1,1).nn.ReLU(),
            nn.Upsample(scale_factor=2), #16000
            nn.Conv1d(1,1).nn.ReLU(),
            nn.Upsample(scale_factor=2), #32000
            nn.Conv1d(1,1).nn.ReLU(),
            nn.Upsample(scale_factor=2), #64000
            nn.Conv1d(1,1).nn.ReLU(),
            )
        # self.up=nn.Upsample(scale_factor=2)
        # self.UpDec=nn.Sequential(
        #     nn.Linear(1024,1024),nn.ReLU(),
        #     nn.Linear(1024,1024)
        # )

        

    def sample_latent(self,x):
        #qualities
        mean1=self.meanL1(x)
        sigma1=torch.sqrt(torch.exp(self.sigmaL1(x)))
        #pitch
        mean2=self.meanL2(x)
        sigma2=torch.sqrt(torch.exp(self.sigmaL2(x)))
        #class
        mean3=self.meanL3(x)
        sigma3=torch.sqrt(torch.exp(self.sigmaL3(x)))        
        self.mean=[mean1,mean2,mean3]
        self.sigma=[sigma1,sigma2,sigma3]
        eps = torch.distributions.normal.Normal(0, 1).sample(sample_shape=sigma1.size())
        z1=mean1+sigma1*Variable(eps,requires_grad=False).cuda()
        z2=mean2+sigma2*Variable(eps,requires_grad=False).cuda()
        z3=mean3+sigma3*Variable(eps,requires_grad=False).cuda()
        return z1,z2,z3
    
    def forward(self,data):
        x=data['audio']
        for L in self.convs:
            x=L(x)
        x=x.view(x.shape[0],128)
        z1,z2,z3=self.sample_latent(x)
        z1=self.qualDec(torch.cat(z1,data['qualities'],dim=1))
        z2=self.pitchDec(torch.cat(z2,data['pitch']),dim=1)
        z3=self.classDec(torch.cat(z3,data['instrument_family']),dim=1)
        z=torch.cat(torch.cat(z1,z2,dim=1),z3,dim=1)
        x=self.jointDec(z)
        x=x.view(x.shape[0],1,2000)
        x=self.UpDec(x)
        return x