 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:54:23 2019

@author: aneesh
"""

import torch
import torch.nn as nn
import torch.nn.init as init

def count_parameters(model):
    '''
    Simple function that counts the total number of parameters in a network.
    Ref: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def weights_init_normal(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.uniform_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            init.uniform_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Linear):
            init.uniform_(m.weight.data, 0.0, 0.02)

def weights_init_xavier(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data, gain=1)
        elif isinstance(m, nn.BatchNorm2d):
            init.uniform_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data, gain=1)
   
def weights_init_kaiming(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif isinstance(m, nn.BatchNorm2d):
            init.uniform_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

def init_weights(net, init_type='normal'):
    if init_type == 'normal':
        weights_init_normal(net)
    elif init_type == 'xavier':
        weights_init_xavier(net)
    elif init_type == 'kaiming':
        weights_init_kaiming(net)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def load_weights(net, path_pretrained_weights):
    pretrained_dict = torch.load(path_pretrained_weights)
    model_dict = net.state_dict()
    
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if 
                      (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
    
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    
    # 3. load the new state dict
    net.load_state_dict(pretrained_dict, strict = False)
    
    # 4. Clear decoder weights
    init_weights(net.up_concat2, init_type='kaiming')
    init_weights(net.up_concat1, init_type='kaiming')
