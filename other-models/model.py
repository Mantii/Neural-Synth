import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class VAE(nn.Module):
	# latent_dim=16
	# encoder_dim=100
	def __init__(self,encoder,decoder):
		super(VAE,self).__init__()
		self.encoder=encoder
		self.decoder=decoder
		self.meanL=nn.Linear(2048,256)
		self.sigmaL=nn.Linear(2048,256)

	def sample_latent(self,h):
		#Construct Mean and Standard Deviation
		mean=self.meanL(h)
		sigma=self.sigmaL(h)
		sigma=torch.exp(sigma)

		#Create epsilon from a gaussian
		eps=torch.Tensor(np.random.normal(0,1,size=256))

		self.z_mean=mean
		self.z_sigma=sigma

		#Reparametrisation Trick
		z=mean+sigma*Variable(eps,requires_grad=False).cuda()
		print(torch.mean(z),torch.std(z),torch.min(z),torch.max(z))
		return z

	def forward(self,state):
		#Encode input
		h=self.encoder(state)
		#Turn into Latent Vector Z
		z=self.sample_latent(h)

		#Reconstruct
		r=self.decoder(z)
		#print(torch.min(r).item(),torch.max(r).item(),torch.mean(r).item())
		return r

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder,self).__init__()
		self.c1=nn.Conv1d(1,128,kernel_size=1,stride=8,dilation=1,padding=0)

		self.c2=nn.Conv1d(128,128,kernel_size=1,stride=16,padding=1)
		self.c3=nn.Conv1d(128,256,kernel_size=1,stride=8,dilation=1,padding=0)
		self.c4=nn.Conv1d(256,256,kernel_size=1,stride=16,padding=1)
			
		self.c5=nn.Conv1d(256,1024,kernel_size=1,stride=4,padding=0)
		self.c6=nn.Conv1d(1024,2048,kernel_size=1,stride=2,padding=0)
		#self.l1=nn.Linear(2048,2048)

		self.b1=nn.BatchNorm1d(128)
		self.b2=nn.BatchNorm1d(128)
		self.b3=nn.BatchNorm1d(256)
		self.b4=nn.BatchNorm1d(256)
		self.b5=nn.BatchNorm1d(1024)
		self.b6=nn.BatchNorm1d(2048)


	def forward(self,x):
		h=torch.tanh(self.b1(self.c1(x)))
		h=torch.tanh(self.b2(self.c2(h)))
		h=torch.tanh(self.b3(self.c3(h)))
		h=torch.tanh(self.b4(self.c4(h)))
		h=torch.tanh(self.b5(self.c5(h)))
		h=F.relu(self.b6(self.c6(h)))
		h=h.view(h.shape[0],1,2048)
		return F.normalize(h)

class Decoder(nn.Module):
	def __init__(self):
		super(Decoder,self).__init__()
		self.l1=nn.Linear(256,2048)

		self.c1=nn.ConvTranspose1d(2048,1024,kernel_size=1,stride=2,output_padding=1)
		self.c2=nn.ConvTranspose1d(1024,256,kernel_size=1,stride=4,output_padding=0)

		self.c3=nn.ConvTranspose1d(256,256,kernel_size=1,stride=16)
		self.c4=nn.ConvTranspose1d(256,128,kernel_size=1,stride=8,dilation=1)


		#self.c5=nn.ConvTranspose1d(128,1,kernel_size=1,stride=1,dilation=0)
		
		self.b3=nn.BatchNorm1d(256)
		self.b4=nn.BatchNorm1d(256)
		self.b5=nn.BatchNorm1d(1024)
		self.b6=nn.BatchNorm1d(2048)
		
	def forward(self,x):
		h=F.relu(self.l1(x))
		h=h.view(h.shape[0],2048,1)
		h=F.relu(self.b5(self.c1(h)))
		h=F.relu(self.b4(self.c2(h)))
		h=3*torch.tanh(self.b3(self.c3(h)))
		h=torch.tanh(self.c4(h))

		h=h.view(h.shape[0],1,65664)[0:h.shape[0],0,0:64000].view(h.shape[0],1,64000)
		return h

if __name__=="__main__":
	encoder=Encoder()
	decoder=Decoder()
	vae=VAE(encoder,decoder)
	print(vae)