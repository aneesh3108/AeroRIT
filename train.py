#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:34:16 2019

@author: aneesh
"""

import os 
import os.path as osp

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import optim

from helpers.augmentations import RandomHorizontallyFlip, RandomVerticallyFlip, \
    RandomTranspose, Compose
from helpers.utils import AeroCLoader, AverageMeter, Metrics, parse_args
from helpers.lossfunctions import cross_entropy2d

from torchvision import transforms

from networks.resnet6 import ResnetGenerator
from networks.segnet import segnet, segnetm
from networks.unet import unet, unetm
from networks.model_utils import init_weights, load_weights

import argparse

# Define a manual seed to help reproduce identical results
torch.manual_seed(3108)

def train(epoch = 0):
    global trainloss
    trainloss2 = AverageMeter()
    
    print('\nTrain Epoch: %d' % epoch)
    
    net.train()

    running_loss = 0.0
    
    for idx, (rgb_ip, hsi_ip, labels) in enumerate(trainloader, 0):
#        print(idx)
        N = hsi_ip.size(0)
        optimizer.zero_grad()
        
        outputs = net(hsi_ip.to(device))
        
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        trainloss2.update(loss.item(), N)
        
        if (idx + 1) %  5 == 0:
            print('[Epoch %d, Batch %5d] loss: %.3f' % (epoch + 1, idx + 1, running_loss / 5))
            running_loss = 0.0
    
    trainloss.append(trainloss2.avg)
    
def val(epoch = 0):
    global valloss
    valloss2 = AverageMeter()
    truth = []
    pred = []
    
    print('\nVal Epoch: %d' % epoch)
    
    net.eval()

    valloss_fx = 0.0
    
    with torch.no_grad():
        for idx, (rgb_ip, hsi_ip, labels) in enumerate(valloader, 0):
    #        print(idx)
            N = hsi_ip.size(0)
            
            outputs = net(hsi_ip.to(device))
            
            loss = criterion(outputs, labels.to(device))
            
            valloss_fx += loss.item()
            
            valloss2.update(loss.item(), N)
            
            truth = np.append(truth, labels.cpu().numpy())
            pred = np.append(pred, outputs.max(1)[1].cpu().numpy())
            
            print("{0:.2f}".format((idx+1)/(len(valset)/100)*100), end = '-', flush = True)
    
    print('VAL: %d loss: %.3f' % (epoch + 1, valloss_fx / (idx+1)))
    valloss.append(valloss2.avg)
    
    return perf(truth, pred)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'AeroRIT baseline evalutions')
    
    ### 0. Config file?
    parser.add_argument('--config-file', default = None, help = 'Path to configuration file')
    
    ### 1. Data Loading
    parser.add_argument('--bands', default = 51, help = 'Which bands category to load \
                        - 3: RGB, 4: RGB + 1 Infrared, 6: RGB + 3 Infrared, 31: Visible, 51: All', type = int)
    parser.add_argument('--hsi_c', default = 'rad', help = 'Load HSI Radiance or Reflectance data?')
    parser.add_argument('--use_augs', action = 'store_false', help = 'Use data augmentations?')
    
    ### 2. Network selections
    ### a. Which network?
    parser.add_argument('--network_arch', default = 'unet', help = 'Network architecture?')
    parser.add_argument('--use_mini', action = 'store_true', help = 'Use mini version of network?')
    
    ### b. ResNet config
    parser.add_argument('--resnet_blocks', default = 6, help = 'How many blocks if ResNet architecture?', type = int)
    
    ### c. UNet configs
    parser.add_argument('--use_SE', action = 'store_true', help = 'Network uses SE Layer?')
    parser.add_argument('--use_preluSE', action = 'store_true', help = 'SE layer uses ReLU or PReLU activation?')
    
    ### Save weights post network config
    parser.add_argument('--network_weights_path', default = None, help = 'Path to save Network weights')
    
    ### Use GPU or not
    parser.add_argument('--use_cuda', action = 'store_true', help = 'use GPUs?')
    
    ### Hyperparameters
    parser.add_argument('--batch-size', default = 100, type = int, help = 'Number of images sampled per minibatch?')
    parser.add_argument('--init_weights', default = 'kaiming', help = "Choose from: 'normal', 'xavier', 'kaiming'")
    parser.add_argument('--learning-rate', default = 1e-4, type = int, help = 'Initial learning rate for training the network?')
    parser.add_argument('--epochs', default = 60, type = int, help = 'Maximum number of epochs?')
    
    ### Pretrained representation present?
    parser.add_argument('--pretrained_weights', default = None, help = 'Path to pretrained weights for network')
    
    args = parse_args(parser)
    print(args)
    
    if args.use_cuda and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    perf = Metrics()
    
    if args.use_augs:
        augs = []
        augs.append(RandomHorizontallyFlip(p = 0.5))
        augs.append(RandomVerticallyFlip(p = 0.5))
        augs.append(RandomTranspose(p = 1))
        augs_tx = Compose(augs)
    else:
        augs_tx = None
        
    tx = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
    
    if args.bands == 3 or args.bands == 4 or args.bands == 6:
        hsi_mode = '{}b'.format(args.bands)
    elif args.bands == 31:
        hsi_mode = 'visible'
    elif args.bands == 51:
        hsi_mode = 'all'
    else:
        raise NotImplementedError('required parameter not found in dictionary')
        
    trainset = AeroCLoader(set_loc = 'left', set_type = 'train', size = 'small', \
                           hsi_sign=args.hsi_c, hsi_mode = hsi_mode,transforms = tx, augs = augs_tx)
    valset = AeroCLoader(set_loc = 'mid', set_type = 'test', size = 'small', \
                         hsi_sign=args.hsi_c, hsi_mode = hsi_mode, transforms = tx)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size = args.batch_size, shuffle = True)
    valloader = torch.utils.data.DataLoader(valset, batch_size = args.batch_size, shuffle = False)
    
    #Pre-computed weights using median frequency balancing    
    weights = [1.11, 0.37, 0.56, 4.22, 6.77, 1.0]
    weights = torch.FloatTensor(weights)
    
    criterion = cross_entropy2d(reduction = 'mean', weight=weights.cuda(), ignore_index = 5)
    
    if args.network_arch == 'resnet':
        net = ResnetGenerator(args.bands, 6, n_blocks=args.resnet_blocks)
    elif args.network_arch == 'segnet':
        if args.mini == True:
            net = segnetm(args.bands, 6)
        else:
            net = segnet(args.bands, 6)
    elif args.network_arch == 'unet':
        if args.use_mini == True:
            net = unetm(args.bands, 6, use_SE = args.use_SE, use_PReLU = args.use_preluSE)
        else:
            net = unet(args.bands, 6)
    else:
        raise NotImplementedError('required parameter not found in dictionary')
   
    init_weights(net, init_type=args.init_weights)
    if args.pretrained_weights is not None:
        load_weights(net, args.pretrained_weights)
        print('Completed loading pretrained network weights')
        
    net.to(device)
    
    optimizer = optim.Adam(net.parameters(), lr = args.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,50])

    trainloss = []
    valloss = []
    
    bestmiou = 0
        
    for epoch in range(args.epochs):
        scheduler.step()
        train(epoch)
        oa, mpca, mIOU, _, _ = val(epoch)
        print('Overall acc  = {:.3f}, MPCA = {:.3f}, mIOU = {:.3f}'.format(oa, mpca, mIOU))
        if mIOU > bestmiou:
            bestmiou = mIOU
            torch.save(net.state_dict(), args.network_weights_path)
