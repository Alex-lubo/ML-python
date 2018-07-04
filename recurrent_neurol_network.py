# coding:utf-8

import os
import sys
import math
import random
import urllib
import zipfile
import collections
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


url = 'http://mattmahoney.net/dc/'

vocabulary_size = 50000
data_index = 0

def download_file(filename, expected_size):
  if not os.path.exists(filename):
    filename, _ = urllib.urlretrieve(url+filename, filename)
  stat_info = os.stat(filename)
  if stat_info.st_size == expected_size:
    print('file exist and verified', filename)
  else:
    print(stat_info.st_size)
    raise Exception(
      'Failed to verify '+ filename + '.You can get it by download tools'
    )
  return filename

def extract_date(filename):
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data

def build_data_set(words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary

def generate_batch(data, batch_size, num_skips, skip_window):
  """
  batch_size: 
  embedding size - 单词的稠密度向量的维度
  num_skips - 对每个目标单词提取的样本数
  skip_window - 单词最远可以联系的距离。
  """
  global data_index
  assert batch_size% num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window
    targets_to_avoid = [ skip_window ]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span-1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i* num_skips + j] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels

def plot_with_labels(low_dim_embs, labels, filename= 'tsne.png'):
  plt.figure(figsize=(18, 18))
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x,y)
    plt.annotate(label,
      xy=(x, y),
      xytext=(5, 2),
      textcoords='offset points',
      ha='right', 
      va='bottom'
    )
  plt.savefig(filename)

def pca_func(dataMat, target_v=0):
  mean_val = np.mean(dataMat, axis=0)
  new_mat = dataMat-mean_val
  cov_mat = np.cov(new_mat, rowvar=0)
  eig_val, eig_vects = np.linalg.eig(cov_mat)
  eig_val_indice = np.argsort(eig_val)
  n_eig_val_indice = eig_val_indice 
  if target_v != 0:
    n_eig_val_indices = eig_val_indice[-1:-(target_v+1):-1]
  n_eig_vects = eig_vects[:, n_eig_val_indice]
  low_data_mat = np.dot(new_mat,mn_eig_vects)
  return low_data_mat


def main(argv=None):
  if argv is None:
    argv = sys.argv
  
  filename = download_file('text8.zip', 31344016)
  words = extract_date(filename)
  print('Data size: ', len(words))

  data, count, dictionary, reverse_dictionary = build_data_set(words)
  del words
  print('Most common words (+unk)', count[:5])
  print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

  batch_size = 128
  embedding_size = 128
  skip_window = 1
  num_skips = 2
  valid_size = 16      #验证单词数
  valid_window = 100   # 从频数最高的100个词中抽取
  valid_examples = np.random.choice(valid_window, valid_size, replace=False) # 随机抽取
  num_sampled = 64     # 负样本的噪声单词的数量

  # network architecture
  graph = tf.Graph()
  with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    with tf.device('/cpu:0'):
      embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
      )
      embed_input = tf.nn.embedding_lookup(embeddings, train_inputs)
      nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0/math.sqrt(embedding_size)))
      nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

      loss = tf.reduce_mean(
        tf.nn.nce_loss(
          weights=nce_weights,
          biases=nce_biases,
          labels=train_labels, 
          inputs=embed_input,
          num_sampled=num_sampled,
          num_classes=vocabulary_size
        )
      )
      optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
      norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
      normalized_embeddings = embeddings / norm
      valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
      similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
      
      init = tf.global_variables_initializer()
      num_steps = 100001
      with tf.Session() as sess:
        init.run()
        print("Iinialized")
        average_loss = 0
        for step in range(num_steps):
          batch_inputs, batch_labels = generate_batch(
            data, batch_size, num_skips, skip_window
          )
          feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
          _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
          average_loss+=loss_val
          if step % 2000 == 0:
            if step > 0:
              average_loss /= 2000
            print ('average loss at step ', step, ': ', average_loss)
            average_loss = 0
          if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
              valid_word = reverse_dictionary[valid_examples[i]]
              top_k = 8
              nearest = (-sim[i, :]).argsort()[1:top_k+1]
              log_str = "Nearest to %s: " % valid_word
              for k in range(top_k):
                close_word = reverse_dictionary[nearest[k]]
                log_str = "%s %s, " % (log_str, close_word)
                print log_str
          finial_embeddings = normalized_embeddings.eval()
          plot_only = 100
          low_dim_embs = pca_func(finial_embeddings[:plot_only, :], 2)
          labels = [reverse_dictionary[i] for i in range(plot_only)]
          plot_with_labels(low_dim_embs, labels)
      
if __name__ == "__main__":
  main()