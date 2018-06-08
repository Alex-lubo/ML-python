# coding=utf-8

"""
多层感知机(MLP)模型
在Softmax Regression的基础上增加了隐藏层，相当于增加了特征抽象的过程，提高匹配和分类的正确率。
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

sess = tf.InteractiveSession()

def input_train():
  """
  add your input procedure to import training data
  """
  batch = mnist.train.next_batch(100)
  return  batch[0], batch[1]

def input_test():
  """
  add your input procedure to import testing data
  """
  return mnist.test.images, mnist.test.labels

# define paramaters
in_uint = 784
h1_uint = 300
class_num = 10

# input data holder variables
x = tf.placeholder(tf.float32, [None, in_uint], name='input_x')
y_ = tf.placeholder(tf.float32, [None, class_num], name='input_y') # labels
keep_prob = tf.placeholder(tf.float32)

# hyper-parameters
W1 = tf.Variable(tf.truncated_normal([in_uint, h1_uint], stddev=0.1), name='weight_lay1')
b1 = tf.Variable(tf.zeros([h1_uint]))
W2 = tf.Variable(tf.truncated_normal([h1_uint, class_num], stddev=0.1), name='weight_lay2')
b2 = tf.Variable(tf.zeros([class_num]))

# define the computation layers and loss function
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden_drop = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden_drop, W2) + b2)
loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))

# define the training function
learning_rate = 0.3
training_steps = 1000
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

tf.global_variables_initializer().run()
for i in range(training_steps):
  batch_x, batch_y = input_train()
  if i % 100 == 0:
    print(
      'step %d accuracy is %g' %(i, accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.}))
    )
  train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.75})

# evalution
test_x, test_y = input_test()
print('total accracy: %g',accuracy.eval(feed_dict={x: test_x, y_:test_y, keep_prob: 1.0}))