# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 21:55:33 2017

@author: dwipr
"""


import numpy as np
import time
import tensorflow as tf

import config

HI = 0


selected_model = config.model
if selected_model == "MODEL_3":
    import model_3ed as Model
elif selected_model == "MODEL_4":
    import model_4ed as Model
elif selected_model == "MODEL_5":
    import model_5ed as Model
else :
    print("Invalid input for model selection!")
def train(n_epochs, trainData, trainLabelOneHot, validationData, validationLabelOneHot):
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
        
    ntrain = len(trainData)
    height = config.height 
    width = config.width
    nrclass = config.nrclass

    #%%
    # Define functions
    x = tf.placeholder(tf.float32, [None, height, width, 3])
    y = tf.placeholder(tf.float32, [None, height, width, nrclass])
    keepprob = tf.placeholder(tf.float32)
    


    weights = weights
    biases = biases
    
    
    #%%
    
    pred = Model.Model(x, weights, biases, keepprob)
    lin_pred = tf.reshape(pred, shape=[-1, nrclass])
    lin_y = tf.reshape(y, shape=[-1, nrclass])
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=lin_pred, labels=lin_y))
    
    # Accuracy
    corr = tf.equal(tf.argmax(y,3), tf.argmax(pred, 3)) 
    accr = tf.reduce_mean(tf.cast(corr, "float"))
    optm = tf.train.AdamOptimizer(0.0001).minimize(cost)
    
    batch_size = 1
    n_epochs = n_epochs
    
    print ("Functions ready")
    #%%
    resumeTraining = config.resumeTraining
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        checkpoint = tf.train.latest_checkpoint("model_chekcpoint/")
        print ("checkpoint: %s" % (checkpoint))
        if resumeTraining == False:
            print ("Memulai pelatihan dari awal.")
        elif  checkpoint:
            print ("Memuat checkpoint..", checkpoint)
            saver.restore(sess, checkpoint)
        else:
            print ("Checkpoint tidak ditemukan. Memulai pelatihan dari awal.")
        
        for epoch_i in range(n_epochs):
            start = int(round(time.time() * 1000))
            trainLoss = []; trainAcc = []
            num_batch = int(ntrain/batch_size)+1
            
            for _ in range(num_batch):

                
                randidx = np.random.randint(ntrain, size=batch_size)
            
                batchData = trainData[randidx]
                batchLabel = trainLabelOneHot[randidx]
                sess.run(optm, feed_dict={x: batchData, y: batchLabel, keepprob: config.dropout_keepprob}) 
                trainLoss.append(sess.run(cost, feed_dict={x: batchData, y: batchLabel, keepprob: 1.}))
                trainAcc.append(sess.run(accr, feed_dict={x: batchData, y: batchLabel, keepprob: 1.}))
            
            # Average loss and accuracy
            trainLoss = np.mean(trainLoss)
            trainAcc = np.mean(trainAcc)
            
            # Run test
            valLoss = sess.run(cost, feed_dict={x: validationData, y: validationLabelOneHot, keepprob: 1.})
            valAcc = sess.run(accr, feed_dict={x: validationData, y: validationLabelOneHot, keepprob: 1.})

            end  = int(round(time.time() * 1000))
            print ("[%02d/%02d] trainLoss: %.4f trainAcc: %.2f valLoss: %.4f valAcc: %.2f time : %.2f" 
                   % (epoch_i, n_epochs, trainLoss, trainAcc, valLoss, valAcc,  float((end-start))/1000))

            if epoch_i == config.n_epoch-1:
                # Save
                saver.save(sess, 'model_checkpoints/', global_step = epoch_i)

    return valLoss, valAcc

