# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 10:42:49 2017

@author: dwipr
"""
import tensorflow as tf
import numpy as np

import data_loading
import config
from model_5ed import Model

model_id = "Model5K5"

import time
import math

from matplotlib.pylab import plt

height = config.height
width = config.width

nrclass = config.nrclass
 
test_imglist, test_data = data_loading.load_data("test")

ntrain = len(test_data)

x = tf.placeholder(tf.float32, [None, height, width, 3])
y = tf.placeholder(tf.float32, [None, height, width, nrclass])
keepprob = tf.placeholder(tf.float32)

nrclass = config.nrclass

# Kernels
ksize = config.ksize
fsize = config.fsize
initstdev = 0.01

initfun = tf.random_normal_initializer(mean=0.0, stddev=initstdev)
# initfun = None
weights = {
    'ce1': tf.get_variable("ce1", shape = [ksize, ksize, 3, fsize], initializer = initfun) ,
    'ce2': tf.get_variable("ce2", shape = [ksize, ksize, fsize, fsize], initializer = initfun) ,
    'ce3': tf.get_variable("ce3", shape = [ksize, ksize, fsize, fsize], initializer = initfun),
    'ce4': tf.get_variable("ce4", shape = [ksize, ksize, fsize, fsize], initializer = initfun),
    'ce5': tf.get_variable("ce5", shape = [ksize, ksize, fsize, fsize], initializer = initfun),
    'cd5': tf.get_variable("cd5", shape = [ksize, ksize, fsize, fsize], initializer = initfun),
    'cd4': tf.get_variable("cd4", shape = [ksize, ksize, fsize, fsize], initializer = initfun),
    'cd3': tf.get_variable("cd3", shape = [ksize, ksize, fsize, fsize], initializer = initfun),
    'cd2': tf.get_variable("cd2", shape = [ksize, ksize, fsize, fsize], initializer = initfun),
    'cd1': tf.get_variable("cd1", shape = [ksize, ksize, fsize, fsize], initializer = initfun),
    'dense_inner_prod': tf.get_variable("dense_inner_prod", shape= [1, 1, fsize, nrclass]
                                       , initializer = initfun) # <= 1x1conv
}
biases = {
    'be1': tf.get_variable("be1", shape = [fsize], initializer = tf.constant_initializer(value=0.0)),
    'be2': tf.get_variable("be2", shape = [fsize], initializer = tf.constant_initializer(value=0.0)),
    'be3': tf.get_variable("be3", shape = [fsize], initializer = tf.constant_initializer(value=0.0)),
    'be4': tf.get_variable("be4", shape = [fsize], initializer = tf.constant_initializer(value=0.0)),
    'be5': tf.get_variable("be5", shape = [fsize], initializer = tf.constant_initializer(value=0.0)),
    'bd5': tf.get_variable("bd5", shape = [fsize], initializer = tf.constant_initializer(value=0.0)),
    'bd4': tf.get_variable("bd4", shape = [fsize], initializer = tf.constant_initializer(value=0.0)),
    'bd3': tf.get_variable("bd3", shape = [fsize], initializer = tf.constant_initializer(value=0.0)),
    'bd2': tf.get_variable("bd2", shape = [fsize], initializer = tf.constant_initializer(value=0.0)),
    'bd1': tf.get_variable("bd1", shape = [fsize], initializer = tf.constant_initializer(value=0.0))
}
        

#%%
pred = Model(x, weights, biases, keepprob)
lin_pred = tf.reshape(pred, shape=[-1, nrclass])
lin_y = tf.reshape(y, shape=[-1, nrclass])
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(lin_pred, lin_y))

# Class label
predmax = tf.argmax(pred, 3)
ymax = tf.argmax(y, 3)

batch_size = 1
n_epochs = 1

print ("Functions ready")

resumeTraining = True
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
        
    start = int(round(time.time() * 1000))
    checkpoint = tf.train.latest_checkpoint("Experiments Checkpoint/"+model_id+"/")
    end  = int(round(time.time() * 1000))
    print ("Load Checkpoint time :", float((end-start))/1000, " seconds")
    
    print ("checkpoint: %s" % (checkpoint))
    if resumeTraining == False:
        print ("Start from scratch")
    elif  checkpoint:   
        print ("Restoring from checkpoint", checkpoint)
        saver.restore(sess, checkpoint)
    else:
        print ("Couldn't find checkpoint to restore from. Starting over.")
    
    for idx in range(ntrain):
        print(idx)
        batchData = test_data[idx:idx+1]
        start = int(round(time.time() * 1000))
        predMaxOut = sess.run(predmax, feed_dict={x: batchData, keepprob:1.})
        plt.imsave(data_loading.dataset_path+"/test_result/"+test_imglist[idx].split("\\")[-1], predMaxOut[0])
        end  = int(round(time.time() * 1000))
        print ("Predict time :", float((end-start))/1000, " seconds")

        
