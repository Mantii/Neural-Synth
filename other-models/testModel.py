#source jupenv/pyenv/bin/activate
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

def conv(inc,outc,ker,strd,pad=0,opad=0,norm=True,trans=False):

	if trans:
		c0=nn.ConvTranspose1d(inc,outc,kernel_size=ker,stride=strd,padding=pad,output_padding=opad)
		c1=nn.ConvTranspose1d(outc,outc,kernel_size=1,stride=1)
		b1=nn.BatchNorm1d(outc)
		a1=nn.Sigmoid()
	else:
		c0=nn.Conv1d(inc,outc,kernel_size=ker,stride=strd,padding=pad)
		c1=nn.Conv1d(outc,outc,kernel_size=1,stride=1)
		b1=nn.BatchNorm1d(outc)
		a1=nn.Sigmoid()
	if norm:
		return [c0,b1,a1,c1,b1,a1]
	else:
		return [c0,a1,c1,a1]


def Encoder_Layers():
	layers=nn.ModuleList([])
	layers.extend(conv(1,128,16,8,1))
	layers.extend(conv(128,256,16,8,1))
	layers.extend(conv(256,512,16,8,1))
	layers.extend(conv(512,1024,16,8,1))
	layers.extend(conv(1024,2048,16,8,pad=1,norm=False))
	return layers
def Decoder_Layers():
	layers=nn.ModuleList([])
	layers.extend(conv(2048,1024,16,8,1,norm=False,trans=True))
	layers.extend(conv(1024,512,16,8,opad=4,trans=True))
	layers.extend(conv(512,256,16,8,1,opad=1,trans=True))
	layers.extend(conv(256,128,16,8,1,opad=1,trans=True))
	layers.extend(conv(128,1,16,8,trans=True))
	return layers

class VAE(nn.Module):

	# encoder_dim=100
	def __init__(self,encoder,decoder):
		super(VAE,self).__init__()
		self.encoder=encoder
		self.decoder=decoder
		self.latent_dim=2048

		self.meanL=nn.Sequential(
			nn.Linear(2048,2048),
			nn.Sigmoid()
			)
		self.sigmaL=nn.Sequential(
			nn.Linear(2048,2048),
			nn.Sigmoid()
			)

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
		#print(torch.mean(z).item(),torch.std(z).item(),torch.min(z).item(),torch.max(z).item())
		return z

	def forward(self,state):
		#Encode input
		h=self.encoder(state)
		#Turn into Latent Vector Z
		z=self.sample_latent(h)

		#Reconstruct
		r=self.decoder(z)
		return r

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder,self).__init__()
		self.layers=Encoder_Layers()
		# self.l1=nn.Sequential(
		# 	nn.Linear(2048,2048),
		# 	nn.Sigmoid()
		# 	)

	def forward(self,x):
		for i in self.layers:
			x=i(x)
		x=x.view(x.shape[0],1,2048)
		return x

class Decoder(nn.Module):
	def __init__(self):
		super(Decoder,self).__init__()
		self.layers=Decoder_Layers()
		# self.fc=nn.Sequential(
		# 	nn.Linear(2048,2048),
		# 	nn.Sigmoid()
		# 	)

	def forward(self,x):
		# x=self.fc(x)
		x=x.view(x.shape[0],2048,1)

		for i in self.layers:
			x=i(x)
		return torch.sigmoid(x)

# if __name__=="__main__":
# 	encoder=Encoder()
# 	decoder=Decoder()
# 	vae=VAE(encoder,decoder)
# 	print(vae)