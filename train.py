import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from option import  opt
from model import SFCSR
from data_utils import TrainsetFromFolder, ValsetFromFolder
from eval import PSNR
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np

import scipy.io as scio  
psnr = []       
out_path = '/media/hdisk/liqiang/hyperSR/result/' +  opt.datasetName + '/'
    
    
def main():

    if opt.show:
        global writer
        writer = SummaryWriter(log_dir='logs') 
       
    if opt.cuda:
        print("=> Use GPU ID: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
		
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True
    
    # Loading datasets
    train_set = TrainsetFromFolder('/media/hdisk/liqiang/hyperSR/train/'+ opt.datasetName + '/' +  str(opt.upscale_factor) + '/')
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)    
    val_set = ValsetFromFolder('/media/hdisk/liqiang/hyperSR/test/' + opt.datasetName + '/' + str(opt.upscale_factor))
    val_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size= 1, shuffle=False)
      
    # Buliding model     
    model = SFCSR(opt)
    criterion = nn.L1Loss() 
    
    if opt.cuda:
      	model = nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
    else:
        model = model.cpu()   
    print('# parameters:', sum(param.numel() for param in model.parameters())) 
                   
    # Setting Optimizer
    optimizer = optim.Adam(model.parameters(),  lr=opt.lr,  betas=(0.9, 0.999), eps=1e-08)    

    # optionally resuming from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)         
            opt.start_epoch = checkpoint['epoch'] + 1 
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))       

    # Setting learning rate
    scheduler = MultiStepLR(optimizer, milestones=[35, 70, 105, 140, 175], gamma=0.5, last_epoch = -1) 


    # Training 
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        scheduler.step()
        print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"])) 
        train(train_loader, optimizer, model, criterion, epoch)         
        val(val_loader, model, epoch)               
        save_checkpoint(model, epoch, optimizer)

def train(train_loader, optimizer, model, criterion, epoch):

    model.train()   
      
    for iteration, batch in enumerate(train_loader, 1):
        input, label = Variable(batch[0]),  Variable(batch[1], requires_grad=False)
        if opt.cuda:
            input = input.cuda()
            label = label.cuda()  
        
        localFeats = []
        for i in range(input.shape[1]):

            if i == 0:
                x = input[:,0:3,:,:]         
                y = input[:,0,:,:]
                new_label = label[:,0,:,:]
        
            elif i == input.shape[1]-1:                 
                 x = input[:,i-2:i+1,:,:]                	   
                 y = input[:,i,:,:]               	
                 new_label = label[:,i,:,:]
            else:
                 x = input[:,i-1:i+2,:,:]                	
                 y = input[:,i,:,:]
                 new_label = label[:,i,:,:] 	
                 	                   	

            SR, localFeats = model(x, y, localFeats, i)    
            localFeats.detach_()
            localFeats = localFeats.detach()
            localFeats = Variable(localFeats.data, requires_grad=False)
            
            loss = criterion(SR, new_label)
            optimizer.zero_grad()
            loss.backward()       
            optimizer.step()         
                                   
        if iteration % 100 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(train_loader), loss.data[0]))

        if opt.show:
            writer.add_scalar('Train/Loss', loss.data[0])  

def val(val_loader, model, epoch):	            

    model.eval()
    val_psnr = 0    
    
    for iteration, batch in enumerate(val_loader, 1):
        input, label = Variable(batch[0], volatile=True),  Variable(batch[1])
        SR = np.ones((label.shape[1], label.shape[2], label.shape[3])).astype(np.float32) 
        
        if opt.cuda:
            input = input.cuda()
        localFeats = []
        for i in range(input.shape[1]):
            if i == 0:
                x = input[:,0:3,:,:]         
                y = input[:,0,:,:]
                new_label = label[:,0,:,:]
        
            elif i == input.shape[1]-1:                 
                 x = input[:,i-2:i+1,:,:]                	   
                 y = input[:,i,:,:]               	
                 new_label = label[:,i,:,:]
            else:
                 x = input[:,i-1:i+2,:,:]                	
                 y = input[:,i,:,:]
                 new_label = label[:,i,:,:] 	
  	
            output, localFeats = model(x, y, localFeats, i)              
               
            SR[i,:,:] = output.cpu().data[0].numpy()  
  	
        val_psnr += PSNR(SR, label.data[0].numpy()) 
    val_psnr = val_psnr / len(val_loader)   
    print("PSNR = {:.3f}".format(val_psnr))       
    if opt.show:
        writer.add_scalar('Val/PSNR',val_psnr, epoch)      
    
        
def save_checkpoint(model, epoch, optimizer):
    model_out_path = "checkpoint/" + "model_{}_epoch_{}.pth".format(opt.upscale_factor, epoch)
    state = {"epoch": epoch , "model": model.state_dict(), "optimizer":optimizer.state_dict()}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")     	
    torch.save(state, model_out_path)

            
if __name__ == "__main__":
    main()
