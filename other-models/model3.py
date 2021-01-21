import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# def conv(in_channels,out_channels,ker,stride,pad=0,opad=0,norm=True,trans=False):
# 	if trans:
# 		c1=nn.ConvTranspose1d(in_channels,out_channels,kernel_size=ker,stride=stride,padding=pad,output_padding=opad)
# 	else:
# 		c1=nn.Conv1d(in_channels,out_channels,kernel_size=ker,stride=stride,padding=pad)
# 	b1=nn.BatchNorm1d(out_channels)
# 	t1=nn.Tanh()
# 	if norm:
# 		layers=[c1,b1,t1]
# 	else:
# 		layers=[c1,t1]
# 	return layers

def conv(in_channels,out_channels,ker,stride,pad=0,opad=0,norm=True,trans=False,activ='tanh'):
	activs={'relu':nn.ReLU(),
	'tanh':nn.Tanh()

	}

	if trans:
		#c1=nn.ConvTranspose1d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,output_padding=0)
		c1=nn.ConvTranspose1d(in_channels,out_channels,kernel_size=ker,stride=stride,padding=pad,output_padding=opad)
		#c3=nn.ConvTranspose1d(out_channels,out_channels,kernel_size=1,stride=1,padding=0,output_padding=0)
	else:
		#c1=nn.Conv1d(in_channels,out_channels,kernel_size=1,stride=1,padding=pad)
		c1=nn.Conv1d(in_channels,out_channels,kernel_size=ker,stride=stride,padding=pad)
		#c3=nn.Conv1d(out_channels,out_channels,kernel_size=1,stride=1,padding=pad)
	b1=nn.BatchNorm1d(out_channels)
	#b2=nn.BatchNorm1d(out_channels)
	#b3=nn.BatchNorm1d(out_channels)
	a1=activs[activ]
	if norm:
		layers=[c1,b1,a1]
	else:
		layers=[c1,a1]

	return layers

def Encoder_Layers():
	layers=nn.ModuleList([])
	layers.extend(conv(1,2,2,2))
	layers.extend(conv(2,4,4,4))
	layers.extend(conv(4,8,4,4))
	layers.extend(conv(8,16,4,4))
	layers.extend(conv(16,32,4,4))
	layers.extend(conv(32,64,4,4,3))
	layers.extend(conv(64,128,4,4))
	layers.extend(conv(128,256,2,2))
	layers.extend(conv(256,512,2,2))
	layers.extend(conv(512,1024,1,2,norm=False))
	return layers
def Decoder_Layers():
	layers=nn.ModuleList([])
	layers.extend(conv(1024,512,2,2,trans=True,norm=False))
	layers.extend(conv(512,256,2,2,trans=True))
	layers.extend(conv(256,128,2,2,trans=True))
	layers.extend(conv(128,64,4,4,trans=True))
	
	layers.extend(conv(64,32,4,4,2,opad=1,trans=True))
	layers.extend(conv(32,16,4,4,trans=True))
	layers.extend(conv(16,8,4,4,trans=True))
	
	
	layers.extend(conv(8,4,4,4,trans=True))
	layers.extend(conv(4,2,4,4,trans=True,norm=False))
	layers.extend(conv(2,1,2,2,trans=True,norm=False))
	return layers

class VAE(nn.Module):

	# encoder_dim=100
	def __init__(self,encoder,decoder):
		super(VAE,self).__init__()
		self.encoder=encoder
		self.decoder=decoder
		self.latent_dim=1024
		self.meanL=nn.Linear(1024,self.latent_dim)
		self.sigmaL=nn.Linear(1024,self.latent_dim)

	def sample_latent(self,h):
		#Construct Mean and Standard Deviation
		mean=self.meanL(h)
		sigma=self.sigmaL(h)
		sigma=torch.exp(sigma)

		#Create epsilon from a gaussian
		eps=torch.Tensor(np.random.normal(0,1,size=self.latent_dim))

		self.z_mean=mean
		self.z_sigma=sigma

		#Reparametrisation Trick
		z=mean+sigma*Variable(eps,requires_grad=False).cuda()
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
		self.layers=Encoder_Layers()
		self.l1=nn.Linear(1024,1024)


	def forward(self,x):
		for i in self.layers:
			x=i(x)
		x=self.l1(x.view(x.shape[0],1,1024))
		return x

class Decoder(nn.Module):
	def __init__(self):
		super(Decoder,self).__init__()
		self.layers=Decoder_Layers()

		self.l1=nn.Linear(1024,1024)

	def forward(self,x):
		x=self.l1(x)
		x=x.view(x.shape[0],1024,1)

		for i in self.layers:
			x=i(x)
		return x

if __name__=="__main__":
	encoder=Encoder()
	decoder=Decoder()
	vae=VAE(encoder,decoder)
	print(vae)