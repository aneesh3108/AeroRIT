#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 22:56:02 2019

@author: aneesh
"""

import torch
import torch.nn as nn

class ChannelSELayer(nn.Module):
    '''
    Squeeze-and-Excitation (SE) block
    Ref: https://github.com/nabsabraham/research-in-pytorch/blob/master/attention/scSE.py
    
    Parameters:
        num_channels    -- number of input channels
        reduction_ratio -- by how much should the channels should be reduced
        act             -- flag to indicate activation between linear layers 
                            in SE (relu vs. prelu)
    
    '''
    def __init__(self, num_channels, reduction_ratio=2, act = 'relu'):
        
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        if act == 'relu':
            self.relu = nn.ReLU()
        elif act == 'prelu':
            self.relu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor