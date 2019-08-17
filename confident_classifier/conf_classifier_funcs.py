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


def process_csv(path, strat_train_idx_frac, seed=42, ood_classes = []):
    gtd = pd.read_csv(path)
    gtd['class_label'] = np.argmax(np.array(gtd.iloc[:,1:len(gtd)])==1., axis=1) 

    tdf = gtd.groupby('class_label', group_keys=False).apply(lambda x: x.sample(frac=strat_train_idx_frac, 
                                                                              random_state=seed))
    tdf['train_idx'] = 1
    gtd = gtd.merge(tdf[['image','train_idx']], how='left', on='image')
    gtd['train_idx'] = gtd['train_idx'].fillna(0)

    if len(ood_classes) > 0:
        gtd.loc[gtd.class_label.isin(ood_classes), 'train_idx'] = 0

    return gtd


# Creating custom data set
class CustomDatasetFromImages(Dataset):
    def __init__(self, data_info, img_folder_path, num_classes=None, transform=None):
        """
        Args:
            data_info (pandas.DataFrame): information about the data
            img_folder_path (string): path to the folder where images are
            nc (int): number of classes for one-hot encoding. if None then use integer labels
            transform: pytorch transforms 
        """
        
        # Read the csv file
        #self.data_info = pd.read_csv(csv_path, header=None)
        self.data_info = data_info
        
        # setting img folder path
        self.img_folder_path = img_folder_path
        
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info['image'])
        
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info['class_label'])
        
        # Number of classes to determine how to create labels
        self.num_classes = num_classes
            
        # Transforms
        self.transform = transform
        self.black_thresh = 5
        self.percentage_black_cutoff = .005
        
        # Calculate len
        self.data_len = len(self.data_info.index)
    
    def _trim_(self, im, tup=(0,0)):
        bg = Image.new(im.mode, im.size, im.getpixel(tup))
        diff = ImageChops.difference(im, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            return im.crop(bbox)

    def __getitem__(self, index):
        # Get image name 
        single_image_name = self.image_arr[index]
        
        # Open image
        img_as_img = Image.open(self.img_folder_path + single_image_name + '.jpg')
        
        # Transform image 
        if self.transform is not None:
            # Crop image if the number of black pixels is > self.percentage_black_cutoff
            prop = 1 - (np.count_nonzero(np.any(img_as_img, axis=-1)) / (img_as_img.size[0] * img_as_img.size[1]))
    
            if prop > self.percentage_black_cutoff:
                tr_img_as_img = self._trim_(self._trim_(img_as_img))
                if tr_img_as_img is None:
                    img_as_img = self.transform(img_as_img)
                else:
                    img_as_img = self.transform(tr_img_as_img)
            else:
                img_as_img = self.transform(img_as_img)
                
        # Get label (class) of the image based on the cropped pandas column
        if self.num_classes is None:
            single_image_label = self.label_arr[index].reshape(-1)
        else:
            single_image_label = torch.nn.functional.one_hot(torch.tensor(self.label_arr[index]), self.num_classes).type(torch.FloatTensor)
        
        return (img_as_img, single_image_label)

    def __len__(self):
        return self.data_len


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class _netD(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=False), 
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16 
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # state size. 1
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1)

class _netG(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16 
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

def Generator(n_gpu, nz, ngf, nc):
    model = _netG(n_gpu, nz, ngf, nc)
    model.apply(weights_init)
    return model

def Discriminator(n_gpu, nc, ndf):
    model = _netD(n_gpu, nc, ndf)
    model.apply(weights_init)
    return model


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            

def initialize_model(model_name, num_classes, feature_extract, input_size=224, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = input_size

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = input_size

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = input_size

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = input_size

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        input_size = input_size

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299
        
    elif 'efficientnet' in model_name:
      model_ft = EfficientNet.from_pretrained(model_name)
      set_parameter_requires_grad(model_ft, feature_extract)
      model_ft._fc = nn.Linear(model_ft._conv_head.out_channels, num_classes)
      input_size = input_size

    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size


