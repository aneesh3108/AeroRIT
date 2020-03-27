#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:11:14 2019

@author: aneesh
"""

import os
import os.path as osp
import torch
import torch.utils.data as data
import cv2
import numpy as np
import yaml
from sklearn.metrics import confusion_matrix

def tensor_to_image(torch_tensor, mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)):
    '''
    Converts a 3D Pytorch tensor into a numpy array for display
    
    Parameters:
        torch_tensor -- Pytorch tensor in format(channels, height, width)
    '''
    for t, m, s in zip(torch_tensor, mean, std):
        t.mul_(s).add_(m)
    
    return np.uint8(torch_tensor.mul(255.0).numpy().transpose(1, 2, 0))
    
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
class Metrics():
    '''
    Calculates all the metrics reported in paper: Overall Accuracy, Average Accuracy,
    mean IOU and mean DICE score
    Ref: https://github.com/rmkemker/EarthMapper/blob/master/metrics.py
    
    Parameters:
        ignore_index -- which particular index to ignore when calculating all values.
                        In AeroRIT, index '5' is the undefined class and hence, the 
                        default value for this function.
    '''
    def __init__(self, ignore_index = 5):
        self.ignore_index = ignore_index
        
    def __call__(self, truth, prediction):
        
        ignore_locs = np.where(truth == self.ignore_index)
        truth = np.delete(truth, ignore_locs)
        prediction= np.delete(prediction, ignore_locs)
        
        self.c = confusion_matrix(truth , prediction)
        return self._oa(), self._aa(), self._mIOU(), self._dice_coefficient(), self._IOU()
            
    def _oa(self):
        return np.sum(np.diag(self.c))/np.sum(self.c)
        
    def _aa(self):
        return np.nanmean(np.diag(self.c)/(np.sum(self.c, axis=1) + 1e-10))
    
    def _IOU(self):
        intersection = np.diag(self.c)
        ground_truth_set = self.c.sum(axis=1)
        predicted_set = self.c.sum(axis=0)
        union =  ground_truth_set + predicted_set - intersection + 1e-10
        
        intersection_over_union = intersection / union.astype(np.float32)
        
        return intersection_over_union
    
    def _mIOU(self):
        intersection_over_union = self._IOU()
        return np.nanmean(intersection_over_union)
    
    def _dice_coefficient(self):
        intersection = np.diag(self.c)
        ground_truth_set = self.c.sum(axis=1)
        predicted_set = self.c.sum(axis=0)
        dice = (2 * intersection) / (ground_truth_set + predicted_set + 1e-10)
        avg_dice = np.nanmean(dice)
        return avg_dice
    
class AeroCLoader(data.Dataset):
    '''
    This function serves as the dataloader for the AeroCampus dataset
    
    Parameters:
        set_loc     -- 'left', 'mid' or 'right' -> indicates which set is to be used
        set_type    -- 'train' or 'test'
        size        -- default 'small' -> 64 x 64.
        hsi_sign    -- 'rad' or 'ref' -> based on which hyperspectral image type is used
        hsi_mode    -- available sampling options are:
                        '3b' -> samples 7, 15, 25 (BGR)
                        '4b' -> samples bands 7, 15, 25, 46 (BGR + IR)
                        '6b' -> samples bands 7, 15, 25, 33, 40, 50 (BGR + 3x IR)
                        'visible' -> samples all visible bands
                        'all' -> samples all 51 bands (visible + infrared)
        transforms  -- transforms for RGB image (default: normalize)
        augs        -- augmentations used (default: horizontal & vertical image flips)
    '''
    
    def __init__(self, set_loc = 'left', set_type = 'train', size = 'small', hsi_sign = 'rad', 
                 hsi_mode = 'all', transforms = None, augs = None):
        
        if size == 'small':
            size = '64'
        else:
            raise Exception('Size not present in the dataset')
        
        self.working_dir = 'Image' + size
        self.working_dir = osp.join('Aerial Data', self.working_dir, 'Data-' + set_loc)
        
        self.rgb_dir = 'RGB'
        self.label_dir = 'Labels'
        self.hsi_sign = hsi_sign
        self.hsi_dir = 'HSI' + '-{}'.format(self.hsi_sign)
        
        self.transforms = transforms
        self.augmentations = augs
        
        self.hsi_mode = hsi_mode
        self.hsi_dict = {
                '3b':[7, 15, 25],
                '4b':[7, 15, 25, 46],
                '6b':[7, 15, 25, 33, 40, 50], 
                'visible':'all 400 - 700 nm',
                'all': 'all 51 bands'}
        
        self.n_classes = 6
        
        with open(osp.join(self.working_dir, set_type + '.txt')) as f:
            self.filelist = f.read().splitlines()
            
    def __getitem__(self, index):
        rgb = cv2.imread(osp.join(self.working_dir, self.rgb_dir, self.filelist[index] + '.tif'))
        rgb = rgb[:,:,::-1]
        
        hsi = np.load(osp.join(self.working_dir, self.hsi_dir, self.filelist[index] + '.npy'))
                
        
        if self.hsi_mode == 'visible':
            hsi = hsi[:,:,0:31]
        elif self.hsi_mode == 'all':
            hsi = hsi
        else:
            bands = self.hsi_dict[self.hsi_mode]
            hsi_temp = np.zeros((hsi.shape[0], hsi.shape[1], len(bands)))
            for i in range(len(bands)):
                hsi_temp[:,:,i] = hsi[:,:,bands[i]]
            hsi = hsi_temp
        
        hsi = hsi.astype(np.float32)
        
        label = cv2.imread(osp.join(self.working_dir, self.label_dir, self.filelist[index] + '.tif'))
        label = label[:,:,::-1]
        
        if self.augmentations is not None:
            rgb, hsi, label = self.augmentations(rgb, hsi, label)
        
        if self.transforms is not None:
            rgb = self.transforms(rgb)
            
            if self.hsi_sign == 'rad':
                hsi = np.clip(hsi, 0, 2**14)/2**14
                hsi = np.transpose(hsi, (2, 0, 1))
                hsi = torch.from_numpy(hsi)
            elif self.hsi_sign == 'ref':
                hsi = np.clip(hsi, 0, 100)/100
                hsi = np.transpose(hsi, (2, 0, 1))
                hsi = torch.from_numpy(hsi)
            
            label = self.encode_segmap(label)
            label = torch.from_numpy(np.array(label)).long()
            
        return rgb, hsi, label
    
    def __len__(self):
        return len(self.filelist)
    
    def get_labels(self):
        return np.asarray(
                [
                        [255, 0, 0],
                        [0, 255, 0],
                        [0, 0, 255],
                        [0, 255, 255],
                        [255, 127, 80],
                        [153, 0, 0],
                        ]
                )
    
    def encode_segmap(self, mask):
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask
    
    def decode_segmap(self, label_mask, plot=False):
        label_colours = self.get_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        
        return np.uint8(rgb)
    
def parse_args(parser):
    '''
    Standard argument parser
    '''
    args = parser.parse_args()
    if args.config_file and os.path.exists(args.config_file):
        data = yaml.safe_load(open(args.config_file))
        delattr(args, 'config_file')
        arg_dict = args.__dict__
#        print (data)
        for key, value in data.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    return args