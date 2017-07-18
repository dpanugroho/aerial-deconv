# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 05:58:09 2017

@author: dwipr
"""
import numpy as np
from train import train
import tensorflow as tf
import data_loading
import config


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
                      validationLabelOneHot=validation_label)
    
    cv_loss+=loss
    cv_acc+=acc
    tf.reset_default_graph()

with open('train_cv_log.txt', 'w') as f:
    f.writelines("Kernel Size : "+str(config.ksize))
    f.writelines("Model : "+config.model)
    f.writelines("CV Loss : "+str(cv_loss/7))
    f.writelines("CV Acc : "+str(cv_acc/7))
    f.writelines("===========================================")
print("CV Loss", cv_loss/7)
print ("CV Acc", cv_acc/7)
