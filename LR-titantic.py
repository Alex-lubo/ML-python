# coding: utf-8

import tensorflow as tf
import os as os

W = tf.Variable(tf.zeros([5, 1]), name="weight")
b = tf.Variable(0., name="bias")
saver = tf.train.Saver()
writer = tf.summary.FileWriter("./supervised_graph", graph=tf.get_default_graph())

def read_csv(batch_size, file_name, record_defaults):
    file_queue = tf.train.string_input_producer([file_name])
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)
    decoded = tf.decode_csv(value, record_defaults=record_defaults)
    return tf.train.shuffle_batch(decoded, batch_size=batch_size, capacity=batch_size*50, min_after_dequeue=batch_size)

def inputs():
  passenger, survived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked = \
    read_csv(100, "./titanic/train.csv", [[0.0],[0.0],[0],[""],[""],[0.0],[0.0],[0.0],[""],[0.0],[""],[""]])
  is_fc = tf.to_float(tf.equal(pclass, [1]))
  is_sc = tf.to_float(tf.equal(pclass, [2]))
  is_tc = tf.to_float(tf.equal(pclass, [3]))
  gender = tf.to_float(tf.equal(sex, ["female"]))

  features = tf.transpose(tf.stack([is_fc, is_sc, is_tc, gender, age]))
  survived = tf.reshape(survived, [100, 1])
  return features, survived

def inference(X):
  return tf.matmul(X, W) + b

def loss(X, Y):
  Y_ = inference(X)
  return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=Y_))  

def train(total_loss):
  learning_rate = 0.008
  return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

def evaluate(sess, X, Y):
  predict = tf.cast(inference(X) > 0.5, tf.float32)
  print 'mst: ', sess.run(tf.reduce_mean(tf.cast(tf.equal(predict, Y), tf.float32)))


with tf.Session() as sess:
  tf.global_variables_initializer().run()
  X, Y = inputs()
  total_loss = loss(X, Y)
  train_op = train(total_loss)
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  loss_summaries = tf.summary.scalar('loss_summaries', total_loss)
  # 实际的训练迭代次数
  training_step = 1000
  for step in range(training_step):
      sess.run([train_op])
      if step % 100 == 0:
          # 保存图
          writer.add_summary(sess.run(loss_summaries), global_step=step)
          print 'loss: ', sess.run(total_loss)
      if step % 1000 == 0:
          # 保存训练记录
          saver.save(sess, './my-model/lr-argrithm', global_step=step)

  # 评估
  evaluate(sess, X, Y)
  coord.request_stop()
  coord.join(threads)
  saver.save(sess, './my-model/lr-argrithm', global_step=training_step)
  writer.flush()
  writer.close()
  sess.close()

