# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# This file was modified from its original form to include statistical reporting
# and to facilitate testing. These code modifications are provided "as is" without
# warranty of any kind.
# ==============================================================================

#!/usr/bin/env python3
r"""Train and export a simple Softmax Regression TensorFlow model.
The model is from the TensorFlow "MNIST For ML Beginner" tutorial. This program
simply follows all its training instructions, and uses TensorFlow SavedModel to
export the trained model with proper signatures that can be loaded by standard
tensorflow_model_server.
Usage: mnist_saved_model.py [--training_iteration=x] [--save_dir export_dir]
"""

import os
import sys

# This is a placeholder for a Google-internal import.

import tensorflow as tf

import mnist_input_data

## MLOps start
from parallelm.mlops import mlops
from parallelm.mlops.stats_category import StatCategory
from parallelm.mlops.stats.table import Table
## MLOps end

tf.app.flags.DEFINE_integer('training_iteration', 1000, 'number of training iterations.')
tf.app.flags.DEFINE_integer('display_step', 100, 'How often to display stats')
tf.app.flags.DEFINE_string('input_cache_dir', '/tmp/mnist_data', 'Input is cached here.')
tf.app.flags.DEFINE_string('save_dir', '/tmp/tf_model', 'Model save base directory.')

FLAGS = tf.app.flags.FLAGS


def main(_):
  if len(sys.argv) < 2 or sys.argv[-1].startswith('-'):
    print('Usage: mnist_export.py [--training_iteration=x] '
          '[--save_dir export_dir] [--display_step=y]')
    sys.exit(-1)
  if FLAGS.training_iteration <= 0:
    print('Please specify a positive value for training iteration.')
    sys.exit(-1)

  # Read the train and test data sets
  mnist = mnist_input_data.read_data_sets(FLAGS.input_cache_dir, one_hot=True)

  ## MLOps start
  # Initialize the mlops library
  mlops.init()

  # Report the feature distribution for the training data
  train_images = mnist.train.images
  mlops.set_data_distribution_stat(train_images)

  # Initialize a table to track training accuracy and cost
  train_table = Table().name("Training Stats").cols(["Accuracy", "Cost"])
  ## MLOps end

  # Create the model
  sess = tf.InteractiveSession()
  serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
  feature_configs = {'x': tf.FixedLenFeature(shape=[784], dtype=tf.float32),}
  tf_example = tf.parse_example(serialized_tf_example, feature_configs)
  x = tf.identity(tf_example['x'], name='x')  # use tf.identity() to assign name
  y_ = tf.placeholder('float', shape=[None, 10])
  w = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  sess.run(tf.global_variables_initializer())
  y = tf.nn.softmax(tf.matmul(x, w) + b, name='y')

  # Set the cost function and optimizer
  cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
  values, indices = tf.nn.top_k(y, 10)
  table = tf.contrib.lookup.index_to_string_table_from_tensor(
      tf.constant([str(i) for i in range(10)]))
  prediction_classes = table.lookup(tf.to_int64(indices))

  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))


  # Train the model
  print('Training model...')
  for i in range(FLAGS.training_iteration):
    batch = mnist.train.next_batch(50)
    _, train_cost, train_acc = sess.run([train_step, cross_entropy, accuracy], feed_dict={x: batch[0], y_: batch[1]})

    # Display stats
    if (i + 1) % FLAGS.display_step == 0 or i + 1 == FLAGS.training_iteration:
      # Report training accuracy and cost

      print("Training. step={}, accuracy={}, cost={}".format(i + 1, train_acc,train_cost))

      # MLOps start
      # multiply by 1 to convert into double
      train_table.add_row("Iterations: {}".format(i+1), [train_acc * 100, train_cost * 1])
      mlops.set_stat(train_table)
      # MLOps end

  print('Done training!')

  # Report final cost and accuracy on test set
  test_cost, test_acc = sess.run([cross_entropy, accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
  print("Testing. accuracy={}, cost={}".format(test_acc, test_cost))

  ## MLOps start
  acc_table = Table().name("Test Accuracy").cols(["Accuracy"])
  acc_table.add_row("Total iterations: {}".format(FLAGS.training_iteration), [test_acc])
  mlops.set_stat(acc_table)

  # Release mlops resources
  mlops.done()
  ## MLOps end

  # Export the trained model so it can be used for inference
  # WARNING(break-tutorial-inline-code): The following code snippet is
  # in-lined in tutorials, please update tutorial documents accordingly
  # whenever code changes.
  export_path = FLAGS.save_dir
  print('Exporting trained model to', export_path)
  builder = tf.saved_model.builder.SavedModelBuilder(export_path)

  # Build the signature_def_map.
  classification_inputs = tf.saved_model.utils.build_tensor_info(
      serialized_tf_example)
  classification_outputs_classes = tf.saved_model.utils.build_tensor_info(
      prediction_classes)
  classification_outputs_scores = tf.saved_model.utils.build_tensor_info(values)

  classification_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
          inputs={
              tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                  classification_inputs
          },
          outputs={
              tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                  classification_outputs_classes,
              tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
                  classification_outputs_scores
          },
          method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))

  tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
  tensor_info_y = tf.saved_model.utils.build_tensor_info(y)

  prediction_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
          inputs={'inputs': tensor_info_x},
          outputs={'outputs': tensor_info_y},
          method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

  legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
  builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.tag_constants.SERVING],
      signature_def_map={
          'predict_images':
              prediction_signature,
          tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
              classification_signature,
      },
      legacy_init_op=legacy_init_op)

  builder.save()

  print('Done exporting!')


if __name__ == '__main__':
  tf.app.run()
