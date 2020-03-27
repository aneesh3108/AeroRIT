#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 22:56:02 2019

@author: aneesh
"""

import torch.nn as nn

class conv2DBatchNormRelu(nn.Module):
    '''
    Standard conv-bn-relu block
    Refs: https://github.com/meetshah1995/pytorch-semseg
    
    Parameters:
        in_channels     -- number of input channels
        out_channels    -- number of output channels
        k_size          -- size of the convolutional kernel: default - 3
        stride          -- length of the stride for cross-correlation
        padding         -- length of zero-padding across all sides
        bias            -- boolean flag to indicate presence of learnable bias
        dilation        -- spacing between kernel with respect to image coords
        is_batchnorm    -- boolean flag to indicate batch-normalization usage
   '''
    def __init__(self, in_channels, out_channels, k_size, stride, padding, bias=True, 
                 dilation=1, is_batchnorm=True):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(out_channels),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_batchnorm:
            self.cbr_unit = nn.Sequential(
                conv_mod, nn.BatchNorm2d(int(out_channels)), nn.ReLU(inplace=True)
            )
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs
    
class segnetUp2(nn.Module):
    '''
    SegNet encoder block with 2 blocks of conv filters with default kernel size of 3
    
    Parameters:
        in_size     -- number of input channels
        out_size    -- number of output channels
    '''
    def __init__(self, in_size, out_size):
        super(segnetUp2, self).__init__()
        
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs

class segnetDown2(nn.Module):
    '''
    SegNet decoder block with 2 blocks of conv filters with default kernel size of 3
    
    Parameters:
        in_size     -- number of input channels
        out_size    -- number of output channels
    '''
    def __init__(self, in_size, out_size):
        super(segnetDown2, self).__init__()
 
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape
    
class segnetUp3(nn.Module):
    '''
    SegNet encoder block with 3 blocks of conv filters with default kernel size of 3
    
    Parameters:
        in_size     -- number of input channels
        out_size    -- number of output channels
    '''
    def __init__(self, in_size, out_size):
        super(segnetUp3, self).__init__()
       
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs
    
class segnetDown3(nn.Module):
    '''
    SegNet decoder block with 3 blocks of conv filters with default kernel size of 3
    
    Parameters:
        in_size     -- number of input channels
        out_size    -- number of output channels
    '''
    def __init__(self, in_size, out_size):
        super(segnetDown3, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape

class segnet(nn.Module):
    '''
    SegNet architecture
    
    Parameters:
        in_channels     -- number of input channels
        out_channels    -- number of output channels
    ''' 
    def __init__(self, in_channels=3, out_channels=21):
        super(segnet, self).__init__()

        self.in_channels = in_channels

        self.down1 = segnetDown2(self.in_channels, 64)
        self.down2 = segnetDown2(64, 128)
        self.down3 = segnetDown3(128, 256)
        self.down4 = segnetDown3(256, 512)

        self.up4 = segnetUp3(512, 256)
        self.up3 = segnetUp3(256, 128)
        self.up2 = segnetUp2(128, 64)
        self.up1 = segnetUp2(64, out_channels)

    def forward(self, inputs):

        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)

        up4 = self.up4(down4, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        final = self.up1(up2, indices_1, unpool_shape1)

        return final

class segnetm(nn.Module):
    '''
    mini SegNet architecture
    
    Parameters:
        in_channels     -- number of input channels
        out_channels    -- number of output channels
    ''' 
    def __init__(self, in_channels=3, out_channels=21):
        super(segnetm, self).__init__()

        self.in_channels = in_channels

        self.down1 = segnetDown2(self.in_channels, 64)
        self.down2 = segnetDown2(64, 128)

        self.up2 = segnetUp2(128, 64)
        self.up1 = segnetUp2(64, out_channels)

    def forward(self, inputs):

        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        
        up2 = self.up2(down2, indices_2, unpool_shape2)
        final = self.up1(up2, indices_1, unpool_shape1)

        return final