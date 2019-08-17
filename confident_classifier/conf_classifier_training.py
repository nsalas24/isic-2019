#!/usr/bin/env python
# coding: utf-8


import time
import os
import copy
import numpy as np
import pandas as pd
from PIL import Image, ImageChops
#from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from efficientnet_pytorch import EfficientNet

from conf_classifier_funcs import process_csv, CustomDatasetFromImages
from conf_classifier_funcs import Generator, Discriminator, set_parameter_requires_grad, initialize_model


# Getting data
df = process_csv(path = 'data/ISIC_2019_Training_GroundTruth.csv',
                  strat_train_idx_frac=0.94,
                  seed=42,
                  ood_classes=[])

df.groupby(['class_label']).agg({'image': 'count'})
df.groupby(['class_label','train_idx']).agg({'image': 'count'}) .groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))

# Data transforms / loaders

data_dir = "data/ISIC_2019_Training_Input/"

# Number of classes in the dataset
num_classes = len(np.unique(df.class_label)) + 1

# Number of epochs to train for 
epochs = 100

input_size = 128

batch_size = 16

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

datasets_dict= {'train': CustomDatasetFromImages(data_info = df[df.train_idx==1.],
                                                 num_classes = None,
                                                 img_folder_path = data_dir,
                                                 transform = data_transforms['train']),
                'val': CustomDatasetFromImages(data_info = df[df.train_idx==0.],
                                               num_classes = None,
                                               img_folder_path = data_dir,
                                               transform = data_transforms['val'])
               }

weights = 1 / np.unique(datasets_dict['train'].data_info['class_label'], return_counts=True)[1]
samples_weight = torch.from_numpy(np.array([weights[c] for c in datasets_dict['train'].data_info['class_label']]))

weighted_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

if weighted_sampler is not None:
    dataloaders_dict = {'train': DataLoader(datasets_dict['train'], batch_size=batch_size, 
                                  num_workers=2, sampler=weighted_sampler),
                    
                    'val': DataLoader(datasets_dict['val'], batch_size=batch_size, 
                                  num_workers=2),
                   
                   'test': DataLoader(datasets_dict['val'], batch_size=1, 
                                  num_workers=2)}
else:
    dataloaders_dict = {'train': DataLoader(datasets_dict['train'], batch_size=batch_size, 
                                  num_workers=2, shuffle=True),
                    
                    'val': DataLoader(datasets_dict['val'], batch_size=batch_size, 
                                  num_workers=2),
                   
                   'test': DataLoader(datasets_dict['val'], batch_size=1, 
                                  num_workers=2)}


# Model parameters
# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "efficientnet-b0"


# Flag for feature extracting. When False, we finetune the whole model, 
#   when True we only update the reshaped layer params
feature_extract = False

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, 
                                        num_classes, 
                                        feature_extract, 
                                        input_size=input_size, 
                                        use_pretrained=True)


# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are 
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)

print("Number of parameters that are being learned:", sum(p.numel() for p in model_ft.parameters() if p.requires_grad))

# GAN parameters
nz = 100
ngf = input_size
ndf = int(ngf/4)
lr = 3e-4
wd = 0.0
decrease_lr = 60
droprate = 0.1
beta = 0.1


print('load GAN')
netG = Generator(1, nz, ngf, 3) # ngpu, nz, ngf, nc
netD = Discriminator(1, 3, ndf) # ngpu, nc, ndf

# Initial setup for GAN
real_label = 1
fake_label = 0
criterion = nn.BCELoss()
fixed_noise = torch.FloatTensor(ndf, nz, 1, 1).normal_(0, 1)

# Put GAN on device
if torch.cuda.is_available():
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    fixed_noise = fixed_noise.cuda()
    
fixed_noise = Variable(fixed_noise)

print('Setup optimizer')
optimizer = optim.Adam(params_to_update, lr=lr) #, weight_decay=wd)
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
#decreasing_lr = list(map(int, decrease_lr.split(',')))

print("Number of parameters that are being learned in Discriminator:", sum(p.numel() for p in netD.parameters() if p.requires_grad))

print("Number of parameters that are being learned in Generator:", sum(p.numel() for p in netG.parameters() if p.requires_grad))

def train(epoch):
    model_ft.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(dataloaders_dict['train']):

        gan_target = torch.FloatTensor(target.size()).fill_(0)
        uniform_dist = torch.Tensor(data.size(0), num_classes).fill_((1./num_classes))

        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
            gan_target, uniform_dist = gan_target.cuda(), uniform_dist.cuda()

        data, target, uniform_dist = Variable(data), Variable(target), Variable(uniform_dist)

        ###########################
        # (1) Update D network    #
        ###########################
        # train with real
        gan_target.fill_(real_label)
        targetv = Variable(gan_target)
        optimizerD.zero_grad()
        output = netD(data)
        errD_real = criterion(output, targetv)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise = torch.FloatTensor(data.size(0), nz, 1, 1).normal_(0, 1).cuda()
        if torch.cuda.is_available():
            noise = noise.cuda()
        noise = Variable(noise)
        fake = netG(noise)
        targetv = Variable(gan_target.fill_(fake_label))
        output = netD(fake.detach())
        errD_fake = criterion(output, targetv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ###########################
        # (2) Update G network    #
        ###########################
        optimizerG.zero_grad()
        # Original GAN loss
        targetv = Variable(gan_target.fill_(real_label))  
        output = netD(fake)
        errG = criterion(output, targetv)
        D_G_z2 = output.data.mean()

        # minimize the true distribution
        KL_fake_output = F.log_softmax(model_ft(fake), dim=0)
        errG_KL = F.kl_div(KL_fake_output, uniform_dist, reduction='batchmean')*num_classes
        generator_loss = errG + beta*errG_KL
        generator_loss.backward()
        optimizerG.step()

        ###########################
        # (3) Update classifier   #
        ###########################
        # cross entropy loss
        optimizer.zero_grad()
        output = F.log_softmax(model_ft(data), dim=0)
        loss = F.nll_loss(output, target.squeeze())

        # KL divergence
        noise = torch.FloatTensor(data.size(0), nz, 1, 1).normal_(0, 1).cuda()
        if torch.cuda.is_available():
            noise = noise.cuda()
        noise = Variable(noise)
        fake = netG(noise)
        KL_fake_output = F.log_softmax(model_ft(fake), dim=0)
        KL_loss_fake = F.kl_div(KL_fake_output, uniform_dist, reduction='batchmean')*num_classes
        total_loss = loss + beta*KL_loss_fake
        total_loss.backward()
        optimizer.step()
        
    _, preds = torch.max(output, 1)
    correct += torch.sum(preds == target.data.squeeze())

    print('Loss: {:.6f}, KL fake Loss: {:.6f}'.format(loss.item(), KL_loss_fake.item()))
    
    #print('\nAccuracy: {}/{} ({:.0f}%)\n'.format(correct, len(dataloaders_dict['train'].dataset),
                                                 #100. * correct / len(dataloaders_dict['train'].dataset)))

def test(epoch):
    model_ft.eval()
    test_loss = 0
    correct = 0
    for data, target in dataloaders_dict['val']:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = F.log_softmax(model_ft(data), dim=0)
        test_loss += F.nll_loss(output, target.squeeze()).item()
        _, preds = torch.max(output, 1) # get the index of the max log-probability
        correct += torch.sum(preds == target.data.squeeze()) 

    test_loss = test_loss
    test_loss /= len(dataloaders_dict['val']) # loss function already averages over batch size
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(dataloaders_dict['val'].dataset),
        100. * correct / len(dataloaders_dict['val'].dataset)))
    
    
for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch, epochs - 1))
    print('-' * 10)
    train(epoch)
    test(epoch)
    if epoch % 10 == 0:
        # do checkpointing
        torch.save(netG.state_dict(), 'models/netG9.pth')
        torch.save(netD.state_dict(), 'models/netD9.pth')
        torch.save(model_ft.state_dict(), 'models/effnetb0_cc9.pth')