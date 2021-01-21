#source jupenv/pyenv/bin/activate
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torchaudio
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from AdvModel import AutoEncoder
from torch.utils.data import DataLoader
from utils import load_dataset,myspec
from os import listdir
import math

from torch.utils.tensorboard import SummaryWriter

tb = SummaryWriter(log_dir='runs/exp1',max_queue=60)

dataloader=load_dataset(samples=5)

trainset=DataLoader(dataloader, batch_size=1240, shuffle=False,drop_last=True, num_workers=4)

def latent_loss(mean,sd,mul):
    mean2=mean*mean
    sd2=sd*sd
    loss=mul*torch.mean(mean2+sd2-torch.log(sd2)-1)
    fll=float(torch.mean(mean2+sd2-torch.log(sd2)-1))
    return loss,fll
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


input_dim = 1
batch_size = 32
num_epochs=100000000
is_training=True
check=False
plot=False
savewav=False
KL=False
saveevery=50
MSE=False


vae = AutoEncoder().cuda()
if MSE:
    crit2 = nn.BCELoss(reduction='mean')
    criterion=nn.MSELoss(reduction='mean')
else:
    criterion = nn.BCELoss(reduction='mean')
    crit2=nn.MSELoss(reduction='mean')


optimizer = optim.Adam(vae.parameters(),lr=1e-4)
# heck = torch.load('../wavenet/batchMSE.tar')
# new={}
# for i in vae.state_dict():
#     try:
#         new[i]=heck[i]
#     except:
#         new[i]=vae.state_dict()[i]
# torch.save({
# 'vae': new,
# 'optimizer_state_dict': optimizer.state_dict(),
# 'epoch':0,#+checkpoint['epoch'],
# 'loss':10,
# 'mul':0
# },'../wavenet/UpStart.tar')

# vae.load_state_dict(new)
if check:
    checkpoint = torch.load('../wavenet/pipesave.tar')
    vae.load_state_dict(checkpoint['vae'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("\nLoaded model checkpoint\nEpoch: ",checkpoint['epoch'],'\nLoss:',checkpoint['loss'])
    mul=checkpoint['mul']
else:
    mul=0

if plot:
    for i,data in enumerate(trainset):
            if i==0:
                x=Variable(data).cuda()
                y=vae(x)
                yp=torch.sigmoid(y).cpu().detach().numpy().reshape(20,62*1024)
                xp=data.reshape(20,62*1024).detach().numpy()
                for a,b in zip(xp,yp):
                    plt.plot(a,'g')
                    plt.plot(b,'r')
                    plt.show()
                    input()
                break



    plt.show()
    exit()
    input()

# for param_group in optimizer.param_groups:
#         param_group['lr'] = 2e-4

print('\nTraining: \n______________________________________')
for s,epoch in enumerate(range(num_epochs)):
    epoch_loss=0
    epoch_ll=0
    miny=0
    maxy=0
    meany=0
    for i,data in enumerate(tqdm(trainset,leave=False)):

        #put x on gpu
        x=Variable(data).cuda()
        vae.zero_grad()
        #put data through layers
        y=vae(x)


        ll,fll=latent_lossp(vae.mean,vae.sigma,mul)
        # if fll>10 and KL==False:
        #   KL=True
        #total loss
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
        miny+=round(float(torch.sigmoid(torch.min(y))),10)
        maxy+=round(float(torch.sigmoid(torch.max(y))),10)
        meany+=round(float(torch.sigmoid(torch.mean(y))),10)
        loss.backward()
        optimizer.step()
    if not is_training:
        break 
    epoch_loss/=(i+1)
    epoch_ll/=(i+1)
    miny/=(i+1)
    maxy/=(i+1)
    meany/=(i+1)
    # if MSE:
    #    # tb.add_scalar('RLossMSE',epoch_loss , epoch+checkpoint['epoch'])
    # else:
    #     #tb.add_scalar('RLoss',epoch_loss , epoch+checkpoint['epoch'])
    # #tb.add_scalar('LLoss',epoch_ll , epoch+checkpoint['epoch'])
    # #tb.add_scalar('mul',mul , epoch+checkpoint['epoch'])
    if math.isnan(epoch_loss) or math.isnan(epoch_ll):
        print('\n\nReceived NaN: Breaking...\n\n')
        break
    if s==0:
        el=epoch_loss
        lls=epoch_ll
        sy=miny
        by=maxy
        my=meany
    # if s%10==0:
    #     print("\nMU:{},{},{},{}  Var:{},{},{},{}\n".format(
    #         str(torch.mean(vae.mean[0]).item())[0:8],
    #         str(torch.mean(vae.mean[1]).item())[0:8],
    #         str(torch.mean(vae.mean[2]).item())[0:8],
    #         str(torch.mean(vae.mean[3]).item())[0:8],
    #         str(torch.mean(vae.sigma[0]).item())[0:8],
    #         str(torch.mean(vae.sigma[1]).item())[0:8],
    #         str(torch.mean(vae.sigma[2]).item())[0:8],
    #         str(torch.mean(vae.sigma[3]).item())[0:8]
    #         ))
    print('Epoch: ',epoch,'\tRCSTRCN: ',str(epoch_loss)[0:8],'\tLL: ',str(epoch_ll)[0:8],
        '\tL2:',str(float(crit2(torch.sigmoid(y),x.view(x.shape[0],1,1024)).item()))[0:8],
        '\tmul: ',mul,'\tKL:',KL)

    #print(vae.weight1[0][0:12])

    if savewav:
        torchaudio.save('output1.wav',denorm(y[0].cpu().detach().view(1,64000),rmin[0],rmax[0]),sample_rate=16000)
        torchaudio.save('output2.wav',denorm(y[1].cpu().detach().view(1,64000),rmin[1],rmax[1]),sample_rate=16000)
        torchaudio.save('output3.wav',denorm(y[2].cpu().detach().view(1,64000),rmin[2],rmax[2]),sample_rate=16000)
        torchaudio.save('output4.wav',denorm(y[3].cpu().detach().view(1,64000),rmin[3],rmax[3]),sample_rate=16000)
    if s%400==0 and s!=0:
        KL= bool(not KL)
    if s%saveevery==0 and s!=0:
        print('Change: ',epoch,'\tRCSTRCN: ',epoch_loss-el,'\tLL:',epoch_ll-lls,
            '\tLoss2:',float(crit2(torch.sigmoid(y),x.view(x.shape[0],1,1024)).item()),
            '\tmean',meany)
        tb.flush()
        torch.save({
            'vae': vae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch':epoch,#+checkpoint['epoch'],
            'loss':epoch_loss,
            'mul':mul

            },'../wavenet/pipesave.tar')
        print("\nModel Saved\n\n")
    if s%saveevery==saveevery-1 and s!=1:
        torch.save({
        'vae': vae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch':epoch,#+checkpoint['epoch'],
        'loss':epoch_loss,
        'mul':mul
        },'../wavenet/pipesaveback.tar')
        #print("\nBackup Saved\n\n")
        # if s!=0:
        #     print('Epoch: ',epoch,'\tEpoch Loss: ',epoch_loss-el,'\tmin:',miny-sy,'\tmax:',maxy-by)

tb.close()

