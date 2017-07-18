# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 08:30:37 2017

@author: dwipr
"""
import tensorflow as tf
import numpy as np

# input : [m, h, w, c]
def Unpooling(inputOrg, size, mask=None):
#    ta = tf.TensorArray(int)

    # m, c, h, w order
    m = size[0]
    h = int(size[1])
    w = int(size[2])
    c = size[3]
    input = tf.transpose(inputOrg, [0, 3, 1, 2])
    x = tf.reshape(input, [-1, 1])
    k = np.float32(np.array([1.0, 1.0]).reshape([1,-1]))
    output = tf.matmul(x, k)
    output = tf.reshape(output,[-1, c, h, w * 2])
    # m, c, w, h
    xx = tf.transpose(output, [0, 1, 3, 2])
    xx = tf.reshape(xx,[-1, 1])
    output = tf.matmul(xx, k)
    # m, c, w, h
    output = tf.reshape(output, [-1, c, w * 2, h * 2])
    output = tf.transpose(output, [0, 3, 2, 1])
    outshape = tf.stack([m, h * 2, w * 2, c])
    if mask != None:
        dense_mask = tf.sparse_to_dense(mask, outshape, output, 0)
        return output, dense_mask
    else:
        return output