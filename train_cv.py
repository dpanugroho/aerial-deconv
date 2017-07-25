# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 05:58:09 2017

@author: dwipr
"""
import numpy as np
from train import train
import tensorflow as tf
import data_loading
import config as config
import pandas as pd

np.random.seed(config.seed)    
trainData, trainLabelOneHot = data_loading.load_data("train")
validationData, validationLabelOneHot = data_loading.load_data("valid")

images = np.vstack((trainData, validationData))
labels = np.vstack((trainLabelOneHot, validationLabelOneHot))

indice = np.arange(len(images))
np.random.shuffle(indice)

n_data = len(images)

# how many cv?
cv = config.cv

cv_split_indices = np.array_split(np.arange(n_data), cv)
 
cv_loss = 0
cv_acc = 0

models = ["MODEL_3"]
ksizes = [5,7]

for model in models:
    for ksize in ksizes: 
        n_epoch = config.n_epoch
        for i in range(len(cv_split_indices)):
            trainIdx = []
            for j in range(len(cv_split_indices)):
                if j != i:
                    trainIdx = np.concatenate((trainIdx, cv_split_indices[j]))
            validationIdx = cv_split_indices[i]
        
            train_images = images[list(np.asarray(trainIdx, dtype='int')),...]
            print(len(train_images))
            validation_images = images[list(np.asarray(validationIdx, dtype='int')),...]
        
            train_label = labels[list(np.asarray(trainIdx, dtype='int')),...]
            validation_label = labels[list(np.asarray(validationIdx, dtype='int')),...]
            

            loss, acc = train(n_epochs=n_epoch, 
                              trainData=train_images,
                              trainLabelOneHot=train_label,
                              validationData=validation_images,
                              validationLabelOneHot=validation_label,
                              selected_model=model,
                              ksize=ksize)
            
            cv_loss+=loss
            cv_acc+=acc
            tf.reset_default_graph()
        print("Config : ",model,ksize)
        print("CV Loss", cv_loss/cv)
        print ("CV Acc", cv_acc/cv)

        log = {'height' : config.height, 
               'width' : config.width,
               'model' : model, 
               'kernel_size' : ksize, 
               'number_of_kernel' : config.fsize, 
               'dropout_keep_probability' : config.dropout_keepprob, 
               'n_epoch' : config.n_epoch, 
               'cv_loss' : cv_loss/cv, 
               'cv_acc' : cv_acc/cv}
        log = pd.DataFrame([log],columns=['height','width','model','kernel_size','number_of_kernel','dropout_keep_probability','n_epoch','cv_loss','cv_acc'])
        log.to_csv('train_log/LOG_'+model+'_K'+str(ksize))