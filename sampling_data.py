#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 17:12:08 2019

@author: aneesh
"""

import os
import os.path as osp
import numpy as np

from skimage import io
from PIL import Image

def create_splits(loc, size_chips, rgb, rad, ref, labels):
    """
    Creates the chips for for train, validation and test sets
    Parameters:
        loc: path to save the created chips. This automatically creates
        subfolders of all underlying domains - RGB, HSI & Labels
        size_chips: size for each chip, preset: 64 x 64
        rgb: path to RGB flight line
        rad: path to Radiance-calibrated flight line
        ref: path to Reflectance-converted flight line
        labels: path to Labels for the flight line
    creates individual chips, with 50% overlap, in the corresponding category and outputs two
    textfiles - train.txt, test.txt. train.txt is a list of all filenames in
    the folder, test.txt contains only non-overlapping chips.
    """
    
    os.makedirs(loc, exist_ok = True)
    os.makedirs(osp.join(loc, 'RGB'), exist_ok = True)
    os.makedirs(osp.join(loc, 'Labels'), exist_ok = True)
    os.makedirs(osp.join(loc, 'HSI-rad'), exist_ok = True)
    os.makedirs(osp.join(loc, 'HSI-ref'), exist_ok = True)
        
    print('Starting chip making now')
    
    trainfile = open(osp.join(loc, 'train.txt'), 'w')
    testfile = open(osp.join(loc, 'test.txt'), 'w')

    x_arr, y_arr, _ = rgb.shape
    
    for xx in range(0, x_arr - size_chips//2, size_chips//2):
        for yy in range(0, y_arr - size_chips//2, size_chips//2):
            
            name = 'image_{}_{}'.format(xx,yy)
            
            rgb_temp = rgb[xx:xx + size_chips, yy:yy + size_chips,:]
            rgb_temp = Image.fromarray(rgb_temp)
            
            hsi_rad_temp = rad[xx:xx + size_chips, yy:yy + size_chips,:]
            hsi_ref_temp = ref[xx:xx + size_chips, yy:yy + size_chips,:]
            
            labels_temp = labels[xx:xx + size_chips, yy:yy + size_chips,:]
            labels_temp = Image.fromarray(labels_temp)
            
            rgb_temp.save(osp.join(loc, 'RGB', name + '.tif'))
            labels_temp.save(osp.join(loc, 'Labels', name + '.tif'))
            np.save(osp.join(loc, 'HSI-rad', name), hsi_rad_temp)
            np.save(osp.join(loc, 'HSI-ref', name), hsi_ref_temp)
            
            trainfile.write("%s\n" % name)
            
            if (xx%size_chips == 0 and yy%size_chips == 0):
                testfile.write("%s\n" % name)

    trainfile.close()
    testfile.close()
    
    print('Stopping chip making now')

if __name__ == "__main__":
    
    folder_dir = osp.join('Aerial Data', 'Collection') #path to full files
    
    image_rgb = io.imread(osp.join(folder_dir, 'image_rgb.tif'))[53:,7:,:]
    
    image_hsi_rad = io.imread(osp.join(folder_dir, 'image_hsi_radiance.tif'))
    image_hsi_rad = np.transpose(image_hsi_rad, [1,2,0])[53:,7:,:]
    
    image_hsi_ref = io.imread(osp.join(folder_dir, 'image_hsi_reflectance.tif'))
    image_hsi_ref = np.transpose(image_hsi_ref, [1,2,0])[53:,7:,:]
    
    labels = io.imread(osp.join(folder_dir, 'image_labels.tif'))[53:,7:,:]
    
###############################################################################
    
    image1_rgb = image_rgb[:,:1728,:]
    image1_rad = image_hsi_rad[:,:1728,:]
    image1_ref = image_hsi_ref[:,:1728,:]
    image1_labels = labels[:,:1728,:]
    
    create_splits(loc = osp.join('Aerial Data', 'Image64', 'Data-left'),
                  size_chips = 64, 
                  rgb = image1_rgb, 
                  rad = image1_rad,
                  ref = image1_ref,
                  labels = image1_labels
                  )
    
###############################################################################
    
    image1_rgb = image_rgb[:,1728:2240,:]
    image1_rad = image_hsi_rad[:,1728:2240,:]
    image1_ref = image_hsi_ref[:,1728:2240,:]
    image1_labels = labels[:,1728:2240,:]
    
    create_splits(loc = osp.join('Aerial Data', 'Image64', 'Data-mid'),
                  size_chips = 64, 
                  rgb = image1_rgb, 
                  rad = image1_rad,
                  ref = image1_ref,
                  labels = image1_labels
                  )
    
###############################################################################
    
    image1_rgb = image_rgb[:,2240:,:]
    image1_rad = image_hsi_rad[:,2240:,:]
    image1_ref = image_hsi_ref[:,2240:,:]
    image1_labels = labels[:,2240:,:]
    
    create_splits(loc = osp.join('Aerial Data', 'Image64', 'Data-right'),
                  size_chips = 64, 
                  rgb = image1_rgb, 
                  rad = image1_rad,
                  ref = image1_ref,
                  labels = image1_labels
                  )
    
###############################################################################

