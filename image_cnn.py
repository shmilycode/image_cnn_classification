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
  def __init(self):
    self.input_features = tf.placeholder(tf.float32, [None,None,3])
    self.input_labels = tf.placeholder(tf.int32)
    self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    pooled_outputs = []
    num_filters = [32, 64]
    filter_sizes = [5, 5]
    num_classes = 100
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
    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 25, 25, 64]
    # Output Tensor Shape: [batch_size, 25 * 25 * 64]
    num_filters_total = num_filters[-1] * np.pow(100/(2*len(filter_sizes)), 2)
    self.h_pool = tf.concat(pooled_outputs, 3)
    self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
  
    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 25 * 25 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=self.h_pool_flat, units=1024, activation=tf.nn.relu)
  
    # Add dropout operation; 0.6 probability that element will be kept
    # Add dropout
    with tf.name_scope("dropout"):
      self.h_drop = tf.nn.dropout(self.dense, self.dropout_keep_prob)
  
    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 100]
    logits = tf.layers.dense(inputs=dropout, units=100)
 
    # Final (unnormalized) scores and predictions
    with tf.name_scope("output"):
        W = tf.get_variable(
            "W",
            shape=[num_filters_total, num_classes],
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
        l2_loss += tf.nn.l2_loss(W)
        l2_loss += tf.nn.l2_loss(b)
        self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
        self.predictions = tf.argmax(self.scores, 1, name="predictions")

    # Calculate mean cross-entropy loss
    with tf.name_scope("loss"):
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_labels)
        self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

    # Accuracy
    with tf.name_scope("accuracy"):
        correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
  
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
      allow_soft_placement=True,
      log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
      image_cnn = ImageCNN()

  # Define Training procedure
  global_step = tf.Variable(0, name="global_step", trainable=False)
  optimizer = tf.train.AdamOptimizer(1e-3)
  grads_and_vars = optimizer.compute_gradients(cnn.loss)
  train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

  # Keep track of gradient values and sparsity (optional)
  grad_summaries = []
  for g, v in grads_and_vars:
      if g is not None:
          grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
          sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
          grad_summaries.append(grad_hist_summary)
          grad_summaries.append(sparsity_summary)
  grad_summaries_merged = tf.summary.merge(grad_summaries)

  # Output directory for models and summaries
  timestamp = str(int(time.time()))
  out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
  print("Writing to {}\n".format(out_dir))

  # Summaries for loss and accuracy
  loss_summary = tf.summary.scalar("loss", cnn.loss)
  acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

  # Train Summaries
  train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
  train_summary_dir = os.path.join(out_dir, "summaries", "train")
  train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

  # Dev summaries
  dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
  dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
  dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

  # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
  checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
  checkpoint_prefix = os.path.join(checkpoint_dir, "model")
  if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
  saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

  # Initialize all variables
  sess.run(tf.global_variables_initializer())

  def train_step(x_batch, y_batch):
      """
      A single training step
      """
      feed_dict = {
        cnn.input_x: x_batch,
        cnn.input_y: y_batch,
        cnn.dropout_keep_prob: 0.6
      }
      _, step, summaries, loss, accuracy = sess.run(
          [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
          feed_dict)
      time_str = datetime.datetime.now().isoformat()
      print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
      train_summary_writer.add_summary(summaries, step)

  def dev_step(x_batch, y_batch, writer=None):
      """
      Evaluates model on a dev set
      """
      feed_dict = {
        cnn.input_x: x_batch,
        cnn.input_y: y_batch,
        cnn.dropout_keep_prob: 1.0
      }
      step, summaries, loss, accuracy = sess.run(
          [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
          feed_dict)
      time_str = datetime.datetime.now().isoformat()
      print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
      if writer:
          writer.add_summary(summaries, step)

  # Generate batches
  batches = reader.batch_iter(
      list(zip(x_train, y_train)), 100, 200)
  # Training loop. For each batch...
  for batch in batches:
      x_batch, y_batch = zip(*batch)
      train_step(x_batch, y_batch)
      current_step = tf.train.global_step(sess, global_step)
      if current_step % 100 == 0:
          print("\nEvaluation:")
          dev_step(x_dev, y_dev, writer=dev_summary_writer)
          print("")
      if current_step % 100 == 0:
          path = saver.save(sess, checkpoint_prefix, global_step=current_step)
          print("Saved model checkpoint to {}\n".format(path))


def main(argv=None):
  train_data,train_labels,eval_data,eval_labels = preprocess()

  train(train_data, train_labels, eval_data, eval_labels)

if __name__ == "__main__":
  tf.app.run()
