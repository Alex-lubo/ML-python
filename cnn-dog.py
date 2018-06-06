# coding:utf-8

import sys
import os
import glob
from itertools import groupby 
from collections import defaultdict
import tensorflow as tf

min_after_dequeue = 10
batch_size = 3
# test converlution kernel
def test_conv2d():
  input_batch = tf.constant([
    [ # image1
      [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], # height
      [[0.1], [1.1], [2.1], [3.1], [4.1], [5.1]],
      [[0.2], [1.2], [2.2], [3.2], [4.2], [5.2]],
      [[0.3], [1.3], [2.3], [3.3], [4.3], [5.3]],
      [[0.4], [1.4], [2.4], [3.4], [4.4], [5.4]],
      [[0.5], [1.5], [2.5], [3.5], [4.5], [5.5]],
    ],
  ])
  kernel = tf.constant([
    [[[0.]], [[0.5]], [[0.]]],# height
    [[[0.]], [[1.0]], [[0.]]],# width
    [[[0.]], [[0.5]], [[0.]]],# channel
  ])
  kernel_canny = tf.constant([
    [
      [[-1., 0., 0.]], [[0., -1., 0.]], [[0., 0., -1.]],
      [[-1., 0., 0.]], [[0., -1., 0.]], [[0., 0., -1.]],
      [[-1., 0., 0.]], [[0., -1., 0.]], [[0., 0., -1.]],
    ],
    [
      [[-1., 0., 0.]], [[0., -1., 0.]], [[0., 0., -1.]],
      [[8., 0., 0.]], [[0., 8., 0.]], [[0., 0., 8.]],
      [[-1., 0., 0.]], [[0., -1., 0.]], [[0., 0., -1.]],
    ],
    [
      [[-1., 0., 0.]], [[0., -1., 0.]], [[0., 0., -1.]],
      [[-1., 0., 0.]], [[0., -1., 0.]], [[0., 0., -1.]],
      [[-1., 0., 0.]], [[0., -1., 0.]], [[0., 0., -1.]],
    ],
  ])
  conv2d = tf.nn.conv2d(input_batch, kernel, strides=[1,3,3,1], padding='SAME')

def save_tf_record_file(sess, dateset, tf_record_location):
  """
  save dateset to tf_record file with label
  parameters:
    sess: tf.Session()
    dateset: dict(list)
      key in the dict is the label of filelist.
    tf_record_location: str
      the saving path of the tf_record file.
  """
  writer = None
  # save every 100 samples info to tf_record in a time to accelerate io efficient.
  current_index = 0
  for breed ,image_filenames in dateset.items():
    for image_filename in image_filenames:
      if current_index % 100 == 0:
        if writer:
          writer.close()
        record_filename = "{tf_record_location}-{current_index}.tfrecords".format(tf_record_location=tf_record_location, current_index=current_index)
        writer = tf.python_io.TFRecordWriter(record_filename)
      current_index += 1
      image_file = tf.read_file(image_filename)
      try:
        image = tf.image.decode_jpeg(image_file)
      except:
        print 'decode jpeg file filed:', image_filename
        continue
      grayscale_image = tf.image.rgb_to_grayscale(image)
      resized_image = tf.image.resize_images(grayscale_image, 250, 151)
      image_bytes = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()
      image_label = breed.encode('utf-8')
      example = tf.train.Example(features=tf.train.Features(feature={
        'lable': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
      }))
      writer.write(example.SerializeToString())
    writer.close()

def prepare_training_data(sess):
  training_dateset = defaultdict(list)
  testing_dateset = defaultdict(list)
  image_filenames = glob.glob('./Images/n02*/*.jpg')
  image_filename_with_breed = map(lambda filename:(filename.split("/")[2], filename), image_filenames)
  for dog_breed, breed_images in groupby(image_filename_with_breed, lambda x:x[0]):
    for i, breed_image in enumerate(breed_images):
      if i % 5 == 0:
        testing_dateset[dog_breed].append(breed_image[1])
      else:
        training_dateset[dog_breed].append(breed_image[1])
  breed_training_count = len(training_dateset[dog_breed])
  breed_testing_count = len(testing_dateset[dog_breed])
  assert round(breed_testing_count/(breed_training_count+breed_testing_count), 2) > 0.18, "Not enough testing images."
  save_tf_record_file(sess, training_dateset, './output/training-images/training-image')
  save_tf_record_file(sess, testing_dateset, './output/testing-images/testing-image')

def load_tf_record_file():
  filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once('./output/training-images/*.tfrecords'))
  tfrecord_reader = tf.TFRecordReader()
  _, tfrecord_serialized = tfrecord_reader.read(filename_queue)
  features = tf.parse_single_example(tfrecord_serialized, features={
    'lable': tf.FixedLenFeature([], tf.string),
    'image': tf.FixedLenFeature([], tf.string)
  })
  tfrecord_image = tf.decode_raw(features['image'], tf.uint8)
  image = tf.reshape(tfrecord_image, [250, 151, 1])
  label = tf.cast(features['label'], tf.string)

  capacity = min_after_dequeue + 3* batch_size
  image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
  return image_batch, label_batch

def combine_input(X):
  return tf.matmul(X, W) + b

def loss(X, Y):
  return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=combine_input(X)))

def train(total_loss):
  learning_rate = 0.1
  return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

def main(argv=None):
  if argv is None:
    argv = sys.argv
  
  with tf.InteractiveSession() as sess:
    if not os.path.exists('./output/training-images', os.R_OK):  
      prepare_training_data(sess)
    images, labels = load_tf_record_file()
    float_image_batch = tf.image.convert_image_dtype(images, tf.float32) # 3*255*151
    conv2d_layer_1 = tf.contrib.layers.converlution2d(
      float_image_batch,
      num_outputs=32,
      kernel_size=(5, 5),
      activation_fn=tf.nn.relu,
      weight_init=tf.random_normal,
      stride=(2, 2),
      trainable=True
    )   # 3*125*76*32
    pool_layer_1 = tf.nn.max_pool(
      conv2d_layer_1,
      ksize=[1, 2, 2, 1],
      strides=[1, 2, 2, 1],
      padding='SAME'
    )   # 3*63*38*32
    conv2d_layer_2 = tf.contrib.layers.converlution2d(
      pool_layer_1,
      num_outputs=64,
      kernel_size=(5, 5),
      activation_fn=tf.nn.relu,
      weight_init=tf.random_normal,
      stride=(1, 1),
      trainable=True
    )   # 3*63*38*64
    pool_layer_2 = tf.nn.max_pool(
      conv2d_layer_2,
      ksize=[1, 2, 2, 1],
      strides=[1, 2, 2, 1],
      padding='SAME'
    )   # 3*32*19*64 
    flattened_layer_2 = tf.reshape(
      pool_layer_2, 
      [batch_size, -1]
    )   # [3, 32*19*63]
    hidden_layer_1 = tf.contrib.layers.full_connected(
      flattened_layer_2,
      512,
      weight_init=lambda i, dtype:tf.truncated_normal([38919, 512], stddev=0.1),
      activation_fn=tf.nn.relu
    )   # [3* 512]
    hidden_layer_1 = tf.nn.dropout(hidden_layer_1, 0.1)
    final_full_connected = tf.contrib.full_connected(
      hidden_layer_1, 
      120,
      weight_init=lambda i,dtype:tf.truncated_normal([512, 120], stddev=0.1)
    )
    all_labels = list(map(lambda c:c.split('/')[1], glob.glob('./Images/*')))
    train_labels = tf.map_fn(lambda l:tf.where(tf.equal(all_labels, l))[0, 0:1][0], labels, dtype=tf.int64)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    train_step = 3000
    for step in range(train_step):
      _, summary = sess.run([train_op, merge_summary])
      train_writer.add_summary(summary, global_step=step)
      if step % 100 == 0:
        print "%d loss: %s" % (step, sess.run(total_loss))



if __name__ == "__main__":
  main()
