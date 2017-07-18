# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 08:43:37 2017

@author: dwipr
"""
import glob
import os
from PIL import Image
import numpy as np
import warnings
import config
from skimage import io, img_as_bool
np.random.seed(149)

# Location of the files

height = config.height
width = config.width

nrclass = config.nrclass

def DenseToOneHot(labels_dense, num_classes):
    # Convert class labels from scalars to one-hot vectors. 
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def load_data(task):
    if task == "test":        
        testPath = os.getcwd() + '/dataset/data_resized/test/sat/'
        imglist = glob.glob(testPath + '/*.png')
        print ("%d Test images" % (len(imglist)))
        
           
        filesLen = len(imglist)
        for (f1, i) in zip(imglist, range(filesLen)):
            # Test image
            img1 = Image.open(f1)
            img1 = img1.resize((width, height))
            rgb  = np.array(img1).reshape(1, height, width, 3)
           
            # Stack images and labels
            if i == 0: 
                data = rgb
            else:
                data = np.concatenate((data, rgb), axis=0)
        return imglist, data
    else:
        if task == "train":
            # Training data
            path1 = os.getcwd() + '/dataset/data_resized/train/sat/'
            path2 = os.getcwd() + '/dataset/data_resized/train/map/'
        elif task == "valid":
            ## Validation data
            path1 = os.getcwd() + '/'+'/dataset/data_resized/valid/sat/'
            path2 = os.getcwd() + '/'+ '/dataset/data_resized/valid/map/'
    

    
        imglist = glob.glob(path1 + '/*.png')
        annotlist = glob.glob(path2 + '/*.png')
        print(task,":")
        print ("%d images" % (len(imglist)))
        print ("%d annotations" % (len(annotlist)))
           
        filesLen = len(imglist)

        for (f1, f2, i) in zip(imglist, annotlist, range(filesLen)):
            # Train image
            img1 = Image.open(f1)
            img1 = img1.resize((width, height))
            rgb  = np.array(img1).reshape(1, height, width, 3)
            # Train  Label
            with warnings.catch_warnings():           
                warnings.simplefilter("ignore")
                img2 = img_as_bool(io.imread(f2))
                label = np.array(img2).reshape(1, height, width, 1)
            # Stack images and labels
            if i == 0: 
                data = rgb
                labels = label
            else:
                data = np.concatenate((data, rgb), axis=0)
                labels = np.concatenate((labels, label), axis=0)
            
        # Onehot-coded label
        labelsOneHot = np.zeros((labels.shape[0], labels.shape[1], labels.shape[2], nrclass))
        for row in range(height):
            for col in range(width):
                single = labels[:, row, col, 0]
                oneHot = DenseToOneHot(single, nrclass) # (367,) => (367, 22)
                labelsOneHot[:, row, col, :] = oneHot
        return data, labelsOneHot

#load_data('test')
