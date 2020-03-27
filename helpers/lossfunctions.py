#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Wed Jul 10 23:48:50 2019

@author: aneesh
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class cross_entropy2d(object):
    '''
    2D cross entropy loss with weighting & ignore_index option.
    Ref: https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loss/loss.py
    
    Parameters:
        reduction       -- specifies if error should be averaged over pixels or 
                            summed or reported as an array (default: mean)
        ignore_index    -- specifies a one-hot value that is ignored and 
                            will not be counted in gradient calculation
        weight          -- weight re-scaling given to each class, usually 
                            calculated by median frequency scaling        
    '''
    def __init__(self, reduction = 'mean', ignore_index = 999, weight = None):
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.weight = weight
    
    def __call__(self, input, target):
        n, c, h, w = input.size()
        nt, ht, wt = target.size()

        # Handle inconsistent size between input and target
        if h != ht and w != wt:  # upsample labels
            input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
    
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(
            input, target, weight=self.weight, reduction=self.reduction, ignore_index=self.ignore_index)
        return loss
    
class GANLoss(nn.Module):
    '''
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    Ref: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    '''
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0, device = 'cuda'):
        '''
        Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        '''
        super(GANLoss, self).__init__()

        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)
            
        self.device = device

    def get_target_tensor(self, prediction, target_is_real):
        '''
        Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        '''

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        '''
        Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        '''
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor.to(self.device))
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss