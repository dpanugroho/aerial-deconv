# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 08:34:09 2017

@author: dwipr
"""
#GPU SELECTION
#gpu_device = '/gpu:0'

# GLOBAL CONSTANTS
height = 400
width = 400

nrclass = 2

fsize = 64
batch_size = 1
seed = 149

# TRAIN CONFIGURATION FOR CROSS-VALIDATION
resumeTraining = False
#model = "MODEL_3"
#ksize = 3

n_epoch = 10

dropout_keepprob = 1

cv = 10