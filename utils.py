from os import listdir
import torchaudio
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import numpy as np
import importlib
import os
import librosa
from six.moves import range  # pylint: disable=redefined-builtin
# import tensorflow.compat.v1 as tf
# from tensorflow.contrib import slim as contrib_slim

def latent_loss(mean,sd,mul):
    mean2=mean*mean
    sd2=sd*sd
    loss=mul*torch.mean(mean2+sd2-torch.log(sd2)-1)
    fll=float(torch.mean(mean2+sd2-torch.log(sd2)-1))
    return loss,fll

def norm(audio):
	rawaudio=torchaudio.transforms.MuLawEncoding()(audio)
	rawaudio=rawaudio.type(torch.FloatTensor)
	rawMin=float(torch.min(rawaudio))
	rawMax=float(torch.max(rawaudio))
	#	rawaudio=(rawaudio-rawMean)/(rawStd)
	rawaudio=(rawaudio-rawMin)/(rawMax-rawMin)
	return rawaudio,rawMin,rawMax

def denorm(audio,min,max):
	rawaudio=(audio*(max-min))+min
	rawaudio=torchaudio.transforms.MuLawDecoding()(rawaudio)
	return rawaudio
def latent_lossp(mean,sd,mul):
    l1,f1=latent_loss(mean[0],sd[0],mul)
    l2,f2=latent_loss(mean[1],sd[1],mul)
    l3,f3=latent_loss(mean[2],sd[2],mul)
    l4,f4=latent_loss(mean[3],sd[3],mul)
    return l1+l2+l3+l4,f1+f2+f3+f4
def upsample(x,rate=2):
    return np.interp(np.linspace(0,len(x),rate*len(x)),np.linspace(0,len(x),len(x)),x)
def downsample(x,rate=2):
    return x[::rate]
def myspec(x):
		spec=specgram(x,
			n_fft=1024,
			hop_length=256,
			mask=True,
			log_mag=True,
			re_im=False,
			dphase=True,
			mag_only=False)
		
		shape=[1]+[513,251,2]
		num_padding = 2**int(np.ceil(np.log(shape[2]) / np.log(2))) - shape[2]
		#reshape
		spgram=spec.reshape(shape)

		#pad
		spgram = np.pad(spgram, [[0, 0], [0, 0], [0, num_padding], [0, 0]])
		#slice
		spgram=spgram[:,0:512,:,:]
		#reshape for torch conv
		spgram=spgram.reshape(2,256,512)
		return spgram

def load_dataset(direc='../wavenet/audio/',samples=2000,savePath=False):
	# Path to Data


	# List to hold Tensors of Raw Audio files
	Raw=[]
	# rmin=[]
	# rmax=[]
	# loop through folders
	for c,d in enumerate(listdir(direc)):
		directory=direc+d+'/'
		print('Collecting Samples In',d)
		for i,file in enumerate(listdir(directory)):
			filepath=directory+file
			# 800 files from each directory
			if i==samples:
				break

			# load file
			waveform, sample_rate = torchaudio.load(filepath,normalization=False)
			waveform,minr,maxr=norm(waveform[0][0:63488])
			waveform=waveform.view(62,1024)
			for samp in waveform:
			
				x=torch.zeros(1025).view(1,1025)
				x[0][0:1024]=samp
				x[0][1024]=c
				# add to list and normalise 
				Raw.append(samp)
			# rmin.append(minr)
			# rmax.append(maxr)


	print("Current Samples: ",len(Raw))

	if savePath is not False:
		# Where to savefile 
		print("DataLoader Proccessing....")
		X=DataLoader(Raw, batch_size=5, shuffle=True,drop_last=True, num_workers=2)
		print("DataLoader Proccessing: Done")

		print("Saving")
		torch.save(X,savePath)

		print("Saved to ",savePath)
		print('Min: ',minbit,'Max: ',maxbit)
	return Raw
def specgram(audio,
						 n_fft=512,
						 hop_length=None,
						 mask=True,
						 log_mag=True,
						 re_im=False,
						 dphase=True,
						 mag_only=False):
	"""Spectrogram using librosa.
	Args:
		audio: 1-D array of float32 sound samples.
		n_fft: Size of the FFT.
		hop_length: Stride of FFT. Defaults to n_fft/2.
		mask: Mask the phase derivative by the magnitude.
		log_mag: Use the logamplitude.
		re_im: Output Real and Imag. instead of logMag and dPhase.
		dphase: Use derivative of phase instead of phase.
		mag_only: Don't return phase.
	Returns:
		specgram: [n_fft/2 + 1, audio.size / hop_length, 2]. The first channel is
			the logamplitude and the second channel is the derivative of phase.
	"""
	if not hop_length:
		hop_length = int(n_fft / 2.)

	fft_config = dict(
			n_fft=n_fft, win_length=n_fft, hop_length=hop_length, center=True)

	spec = librosa.stft(audio, **fft_config)

	if re_im:
		re = spec.real[:, :, np.newaxis]
		im = spec.imag[:, :, np.newaxis]
		spec_real = np.concatenate((re, im), axis=2)

	else:
		mag, phase = librosa.core.magphase(spec)
		phase_angle = np.angle(phase)

		# Magnitudes, scaled 0-1
		if log_mag:
			mag = (librosa.power_to_db(
					mag**2, amin=1e-13, top_db=120., ref=np.max) / 120.) + 1
		else:
			mag /= mag.max()

		if dphase:
			#  Derivative of phase
			phase_unwrapped = np.unwrap(phase_angle)
			p = phase_unwrapped[:, 1:] - phase_unwrapped[:, :-1]
			p = np.concatenate([phase_unwrapped[:, 0:1], p], axis=1) / np.pi
		else:
			# Normal phase
			p = phase_angle / np.pi
		# Mask the phase
		if log_mag and mask:
			p = mag * p
		# Return Mag and Phase
		p = p.astype(np.float32)[:, :, np.newaxis]
		mag = mag.astype(np.float32)[:, :, np.newaxis]
		if mag_only:
			spec_real = mag[:, :, np.newaxis]
		else:
			spec_real = np.concatenate((mag, p), axis=2)
	return spec_real


def inv_magphase(mag, phase_angle):
	phase = np.cos(phase_angle) + 1.j * np.sin(phase_angle)
	return mag * phase


def griffin_lim(mag, phase_angle, n_fft, hop, num_iters):
	"""Iterative algorithm for phase retrieval from a magnitude spectrogram.
	Args:
		mag: Magnitude spectrogram.
		phase_angle: Initial condition for phase.
		n_fft: Size of the FFT.
		hop: Stride of FFT. Defaults to n_fft/2.
		num_iters: Griffin-Lim iterations to perform.
	Returns:
		audio: 1-D array of float32 sound samples.
	"""
	fft_config = dict(n_fft=n_fft, win_length=n_fft, hop_length=hop, center=True)
	ifft_config = dict(win_length=n_fft, hop_length=hop, center=True)
	complex_specgram = inv_magphase(mag, phase_angle)
	for i in range(num_iters):
		audio = librosa.istft(complex_specgram, **ifft_config)
		if i != num_iters - 1:
			complex_specgram = librosa.stft(audio, **fft_config)
			_, phase = librosa.magphase(complex_specgram)
			phase_angle = np.angle(phase)
			complex_specgram = inv_magphase(mag, phase_angle)
	return audio


def ispecgram(spec,
							n_fft=512,
							hop_length=None,
							mask=True,
							log_mag=True,
							re_im=False,
							dphase=True,
							mag_only=True,
							num_iters=1000):
	"""Inverse Spectrogram using librosa.
	Args:
		spec: 3-D specgram array [freqs, time, (mag_db, dphase)].
		n_fft: Size of the FFT.
		hop_length: Stride of FFT. Defaults to n_fft/2.
		mask: Reverse the mask of the phase derivative by the magnitude.
		log_mag: Use the logamplitude.
		re_im: Output Real and Imag. instead of logMag and dPhase.
		dphase: Use derivative of phase instead of phase.
		mag_only: Specgram contains no phase.
		num_iters: Number of griffin-lim iterations for mag_only.
	Returns:
		audio: 1-D array of sound samples. Peak normalized to 1.
	"""
	if not hop_length:
		hop_length = n_fft // 2

	ifft_config = dict(win_length=n_fft, hop_length=hop_length, center=True)

	if mag_only:
		mag = spec[:, :, 0]
		phase_angle = np.pi * np.random.rand(*mag.shape)
	elif re_im:
		spec_real = spec[:, :, 0] + 1.j * spec[:, :, 1]
	else:
		mag, p = spec[:, :, 0], spec[:, :, 1]
		if mask and log_mag:
			p /= (mag + 1e-13 * np.random.randn(*mag.shape))
		if dphase:
			# Roll up phase
			phase_angle = np.cumsum(p * np.pi, axis=1)
		else:
			phase_angle = p * np.pi

	# Magnitudes
	if log_mag:
		mag = (mag - 1.0) * 120.0
		mag = 10**(mag / 20.0)
	phase = np.cos(phase_angle) + 1.j * np.sin(phase_angle)
	spec_real = mag * phase

	if mag_only:
		audio = griffin_lim(
				mag, phase_angle, n_fft, hop_length, num_iters=num_iters)
	else:
		audio = librosa.core.istft(spec_real, **ifft_config)
	return np.squeeze(audio / audio.max())

def trainVAE(data,model,optim,epochs=10,criterion=torch.nn.BCELoss(),latent_loss=None,mul=0):



	for epoch in range(epochs):
		epoch_ll=0
		epoch_rl=0
		for i,d in enumerate(tqdm(data,leave=False)):
			x=Variable(d).cuda()
			y=model(x)
			loss=criterion(y,x)+latent_loss(model.mean,model.sigma,mul)
			epoch_ll+=float(latent_loss(model.mean,model.sigma,mul))
			epoch_rl+=float(loss-latent_loss(model.mean,model.sigma,mul))
			loss.backwards()
			optim.step()
		if mul<1:
			mul+=0.1
			epoch_ll/=(i+1)
			epoch_rl/=(i+1)
			print('Epoch: {}\tRL: {}\tLL: {}'.format(epoch,epoch_rl,epoch_ll))

	return model,optim



