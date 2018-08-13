# coding=utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

n_inputs=28
max_time=28
lstm_size = 200
n_class = 10
batch_size = 50
n_batch = mnist.train.num_examples//batch_size
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

weights = tf.Variable(tf.truncated_normal([lstm_size, n_class], stddev = 0.1))
bias = tf.Variable(tf.constant(0.1, shape=[n_class]))

def LSTM(X, weight, bias):
  inputs = tf.reshape(X, [-1, max_time, n_inputs])
  lstm_cell = rnn.BasicLSTMCell(lstm_size)
  outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
  results = tf.nn.softmax(tf.matmul(final_state[1], weights)+bias)
  return results

prediction = LSTM(x,weights, bias)
crossentropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
train_step = tf.train.AdamOptimizer(1e-4).minimize(crossentropy)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1)), tf.float32))
init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)
  for epoch in range(70):
    for batch in range(batch_size):
      batch_x, batch_y = mnist.train.next_batch(batch_size)
      sess.run(train_step, feed_dict={x: batch_x, y:batch_y})
    acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
    print('iter', epoch, 'accuracy', acc)
