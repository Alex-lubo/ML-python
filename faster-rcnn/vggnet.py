# coding=utf-8
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

# input = ..
# net = slim.conv2d(input, 64, [3, 3], scope='conv1_1')
# net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
# net = slim.max_pool2d(net, [2, 2], scope="poll2")
pretrain_path = "./vgg16.ckpt"
reader = tf.train.NewCheckpointReader(pretrain_path)
var_to_shape_map = reader.get_variable_to_shape_map()
var = tf.get_variable(name='vgg_16/conv1/conv1_1/weights', shape=[3,3,3,64], initializer=tf.truncated_normal_initializer(stddev=0.04, dtype=tf.float32), dtype=tf.float32)
pretrain_collection = [var]
saver1 = tf.train.Saver(pretrain_collection, write_version=tf.train.SaverDef.V2)
init = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)
  saver1.restore(sess, pretrain_path)
  print 'a----', var_to_shape_map
  print(var)


