import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torchaudio
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from linModel import AutoEncoder
from torch.utils.data import DataLoader
from utils import load_dataset,myspec,latent_loss
from os import listdir
import math

dataloader=load_dataset(samples=5)





batch_size = 32
num_epochs=100000000
learning_rate=1e-3


trainset=DataLoader(dataloader, batch_size=batch_size, shuffle=False,drop_last=True, num_workers=4)


KL=False 
saveevery=50
MSE=False
check=False
checkpoint_dir='../wavenet/pipesave.tar'

#Initialise network and put it on GPU
vae = AutoEncoder().cuda()
# Whether to use MSE or BCE for the loss function and initalising to track the other loss function.
if MSE:
    crit2 = nn.BCELoss(reduction='mean')
    criterion=nn.MSELoss(reduction='mean')
else:
    criterion = nn.BCELoss(reduction='mean')
    crit2=nn.MSELoss(reduction='mean')

#Initialise Optimiser
optimizer = optim.Adam(vae.parameters(),lr=learning_rate)

# Load a checkpoint if there is one.
if check:
    checkpoint = torch.load(checkpoint_location)
    vae.load_state_dict(checkpoint['vae'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("\nLoaded model checkpoint\nEpoch: ",checkpoint['epoch'],'\nLoss:',checkpoint['loss'])
    mul=checkpoint['mul']
else:
    mul=0



print('\nTraining: \n______________________________________')
for s,epoch in enumerate(range(num_epochs)):
    epoch_loss=0
    epoch_ll=0
    for i,data in enumerate(tqdm(trainset,leave=False)):

        #put x on gpu
        x=Variable(data).cuda()
        vae.zero_grad()
        #put data through layers
        y=vae(x)

        # Get Latent Loss with the mul multiplier which adjusts how impactful to the overall loss function the latent loss is.
        ll,fll=latent_loss(vae.mean,vae.sigma,mul)

        #It is good to slowly ramp up how impactful the KL divergence is as otherwise the model will just minimise KL immedietly 
        # and cause the latent space to be equal to 0
        
        if KL:
            if mul<0.1:
                mul+=1e-5
            else:
                mul=0.1
            loss=criterion(torch.sigmoid(y),x.view(x.shape[0],1,1024))+ll
        else:
            mul=0
            loss=criterion(torch.sigmoid(y),x.view(x.shape[0],1,1024))+ll



        epoch_loss+=(float(loss.item())-float(ll.item()))
        epoch_ll+=fll
        loss.backward()
        optimizer.step()

    epoch_loss/=(i+1)
    epoch_ll/=(i+1)

    if math.isnan(epoch_loss) or math.isnan(epoch_ll):
        print('\n\nReceived NaN: Breaking...\n\n')
        break
    if s==0:
        el=epoch_loss
        lls=epoch_ll

    print('Epoch: ',epoch,'\tRCSTRCN: ',str(epoch_loss)[0:8],'\tLL: ',str(epoch_ll)[0:8],
        '\tL2:',str(float(crit2(torch.sigmoid(y),x.view(x.shape[0],1,1024)).item()))[0:8],
        '\tmul: ',mul,'\tKL:',KL)


    #Sometimes it is good to turn KL off completly so the model can learn more.
    if s%400==0 and s!=0:
        KL= bool(not KL)
    #How often the model makes a checkpoint
    if s%saveevery==0 and s!=0:
        print('Change: ',epoch,'\tRCSTRCN: ',epoch_loss-el,'\tLL:',epoch_ll-lls,
            '\tLoss2:',float(crit2(torch.sigmoid(y),x.view(x.shape[0],1,1024)).item()),
            '\tmean',meany)
        torch.save({
            'vae': vae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch':epoch,#+checkpoint['epoch'],
            'loss':epoch_loss,
            'mul':mul

            },'../wavenet/pipesave.tar')
        print("\nModel Saved\n\n")
    #backup checkpoint
    if s%saveevery==saveevery-1 and s!=1:
        torch.save({
        'vae': vae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch':epoch,#+checkpoint['epoch'],
        'loss':epoch_loss,
        'mul':mul
        },'../wavenet/pipesaveback.tar')


