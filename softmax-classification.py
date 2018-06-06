# coding:utf-8
import tensorflow as tf
import numpy as np
import pandas as pd

W = tf.Variable(tf.zeros([4,3], dtype=tf.float64), name='weigth')
b = tf.Variable(tf.zeros([3], dtype=tf.float64), name='bias')

def inputs():
  column_names = ['sepel_length', 'sepel_width', 'petal_length','petal_width', 'label']
  date_type = {
    'sepel_length':np.float, 'sepel_width':np.float, 'petal_length':np.float,'petal_width':np.float, 'label':np.int32
  }
  train_df = pd.read_csv(filepath_or_buffer = './Iris/iris_training.csv', names=column_names, dtype=date_type,header=0)
  test_df = pd.read_csv(filepath_or_buffer = './Iris/iris_test.csv', names=column_names, dtype=date_type, header=0)
  train_features, train_labels = train_df, train_df.pop('label')
  test_features, test_labels = test_df, test_df.pop('label')
  return (train_features.values, train_labels.values), (test_features.values, test_labels.values)

def combine_input(X):
  return tf.matmul(X, W) + b

def inference(X):
  return tf.nn.softmax(combine_input(X))

def loss(X, Y):
  return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=combine_input(X)))

def train(total_loss):
  learning_rate = 0.1
  return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

def evaluate(sess, X, Y):
  predict = tf.cast(tf.argmax(inference(X), 1), tf.int32)
  print 'evaluate ', sess.run(tf.reduce_mean(tf.cast(tf.equal(predict, Y), tf.float32)))

with tf.Session() as sess:
  init = tf.global_variables_initializer()
  sess.run(init)
  train_writer = tf.summary.FileWriter(logdir='./softmax-classification', graph=sess.graph)
  (train_x, train_y),(test_x, test_y) = inputs()
  total_loss = loss(train_x, train_y)
  tf.summary.scalar('loss_summaries', total_loss)
  merge_summary = tf.summary.merge_all()
  train_op = train(total_loss)
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  train_step = 3000
  for step in range(train_step):
    _, summary = sess.run([train_op, merge_summary])
    train_writer.add_summary(summary, global_step=step)
    if step % 100 == 0:
      print "%d loss: %s" % (step, sess.run(total_loss))

  evaluate(sess, test_x, test_y)
  train_writer.flush()
  train_writer.close()
  sess.close()