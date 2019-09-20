"""Convolutional Neural Network Estimator for Image classification, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import reader

tf.logging.set_verbosity(tf.logging.INFO)


class ImageCNN(object):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # images are 100x100 pixels, and have 3 color channel
  def __init(self, features, labels):
    self.input_features = tf.placeholder(tf.float32, [None,None,3])
    self.input_labels = tf.placeholder(tf.int32)
    pooled_outputs = []
    num_filters = [32, 64]
    filter_sizes = [5, 5]
    for i, filter_size in enumerate(filter_sizes)):
      num_filter = num_filters[i]
      with tf.name_scope("conv-maxpool-%s" % filter_size)
        #Convolution layer
        filter_shape = [filter_size, fileter_size, 1, num_fileter
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_filter), name="b")

        # Convolutional Layer
        # Computes 32 features using a 5x5 filter with ReLU activation.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, 100, 100, 3]
        # Output Tensor Shape: [batch_size, 100, 100, num_filter]
        conv = tf.nn.conv2d(
            self.input_features,
            W,
            padding="same",
            name="conv")
        # Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

        # Pooling Layer
        # First max pooling layer with a 2x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 100, 100, num_filter]
        # Output Tensor Shape: [batch_size, 50, 50, num_filter]
        pooled = tf.nn.max_pool2d(
            h,
            ksize=[2, 2],
            strides=2,
            name="pool")
        pooled_outputs.append(pooled)

    # Combine all the pooled features
    num_filters_total = num_filters[-1] * len(filter_sizes)
    self.h_pool = tf.concat(pooled_outputs, 3)
    self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 25, 25, 64]
    # Output Tensor Shape: [batch_size, 25 * 25 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 25 * 25 * 64])
  
    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 25 * 25 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  
    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
  
    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 100]
    logits = tf.layers.dense(inputs=dropout, units=100)
  
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
  
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
      train_op = optimizer.minimize(
          loss=loss,
          global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
  
    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def preprocess():
  # Load training and eval data
  samples,labels,categories = reader.load_data("../data_set/train2014/", "../data_set/annotations/instances_train2014.json")
  print("Load data finish!!")
  state = np.random.get_state()
  np.random.shuffle(samples)
  np.random.set_state(state)
  np.random.shuffle(labels)
  train_data_len = int(0.8*len(labels))
  train_data = np.asarray(samples[0:train_data_len], dtype=np.float32)
  train_labels = np.asarray(labels[0:train_data_len], dtype=np.int32)
  eval_data = np.asarray(samples[train_data_len:], dtype=np.float32)
  eval_labels = np.asarray(labels[train_data_len:], dtype=np.int32)
  return  train_data,train_labels,eval_data,eval_labels


def train(train_data, train_label, eval_data, eval_labels):
  with tf.Graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
      image_cnn = Image
  # Create the Estimator
  image_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="./image_convnet_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  image_classifier.train(
      input_fn=train_input_fn,
      steps=20000,
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
      x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
  eval_results = image_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

def main(argv=None):
  train_data,train_labels,eval_data,eval_labels = preprocess()

  train(train_data, train_labels, eval_data, eval_labels)

if __name__ == "__main__":
  tf.app.run()
