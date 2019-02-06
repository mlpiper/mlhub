"""
This code loads a saved tensorflow model and predicts on new samples.
The saved model in this case is a retrained model on a pre-trained model.
This is the inference part of transfer learning.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys
import time

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from os import listdir
from os.path import isfile, join
from random import randint

from parallelm.mlops import mlops
from parallelm.mlops.predefined_stats import PredefinedStats
from parallelm.mlops.stats.bar_graph import BarGraph

def add_jpeg_decoding(input_height, input_width, input_depth):
  """Adds operations that perform JPEG decoding and resizing to the graph..

  Args:
    input_height: Expected input image height.
    input_width: Expected input image width.
    input_depth: Expected input image depth, typically 3 for RGB images.

  Returns:
    Tensors for the node to feed JPEG data into, and the output of the
      preprocessing steps.
  """

  jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
  decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)

  # Convert from full range of uint8 to range [0,1] of float32.
  decoded_image_as_float = tf.image.convert_image_dtype(decoded_image,
                                                        tf.float32)
  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
  resize_shape = tf.stack([input_height, input_width])
  resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
  resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                           resize_shape_as_int)
  return jpeg_data, resized_image



def main(_):
  with tf.Session() as sess:
    # Set the tag to load a tensorflow model
    tag_set = "serve"
    # Load the model
    tf.saved_model.loader.load(sess, [tag_set], FLAGS.model_dir)
    # Initialize and load the input and output tensors of the graph
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("input_image:0")
    y = graph.get_tensor_by_name("final_result:0")


    # Set up the image decoding sub-graph.
    shape = x.get_shape().as_list()
    jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(shape[1], shape[2], shape[3])

    # Load input data
    labels = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    images = {}
    count = {}
    for label in labels:
      image_dir = os.path.join(FLAGS.image_base_dir, label)
      images[label] = []
      for f in listdir(image_dir):
        fd = os.path.join(image_dir, f)
        if isfile(fd):
          images[label].append(fd)
      count[label] = 0


    # Perform predictions
    output_file = open(FLAGS.output_file, "w")
    output_file.write("confidence, prediction, label\n")

    correct_predictions = 0
    duration = 0

    # TODO: this would run faster in mini-batches
    for i in range(0, FLAGS.total_predictions):
      label_index = randint(0, len(labels) - 1)
      label = labels[label_index]
      which_file = randint(0, len(images[label]) - 1)
      file = images[label][which_file]
      image_data = tf.gfile.FastGFile(file, 'rb').read()

      start_time = time.time()
      resized_input_values = sess.run(decoded_image_tensor, {jpeg_data_tensor: image_data})
      # Print the prediction results
      inference = sess.run(y, {x:resized_input_values})
      end_time = time.time()
      duration += end_time - start_time
      prediction = np.argmax(inference[0])
      confidence = inference[0, prediction]
      predicted_label = labels[prediction]
      output_file.write("{}, {}, {}\n".format(confidence, predicted_label, label))
      count[predicted_label] += 1
      if predicted_label == label:
        correct_predictions += 1

    output_file.close()

    # Report statistics for this run
    mlops.init()

    prediction_hist = BarGraph().name("categories").cols(labels)
    prediction_hist.data([count['daisy'], count['dandelion'], count['roses'], count['sunflowers'], count['tulips']])
    mlops.set_stat(prediction_hist)

    mlops.set_stat("correct percent", correct_predictions * 100.0 / FLAGS.total_predictions)
    mlops.set_stat("batch seconds", duration)
    mlops.set_stat(PredefinedStats.PREDICTIONS_COUNT, FLAGS.total_predictions)

    mlops.done()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--image_base_dir',
      type=str,
      default='/data-lake/tensorflow/flower_photos',
      help='Base directory for the input images.'
  )
  parser.add_argument(
      '--model_dir',
      type=str,
      default='/data-lake/tensorflow/tf_image_model',
      help='Where to read the trained graph.'
  )
  parser.add_argument(
      '--total_predictions',
      type=int,
      default=100,
      help='Number of predictions to make.'
  )
  parser.add_argument(
      '--output_file',
      type=str,
      default='/tmp/image_predictions',
      help='File to write output.'
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
