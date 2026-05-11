# model_trainer.py
# ML Lifecycle Stage: Training
# Responsible for building the final classification layer on top of the
# frozen Inception v3 bottleneck, running the training loop, and logging
# summaries to TensorBoard.
# No data loading, no bottleneck caching, no evaluation, no export lives here.

import time
from datetime import datetime

import tensorflow as tf

from bottleneck_cache import get_random_cached_bottlenecks
from augmentation import get_random_distorted_bottlenecks

BOTTLENECK_TENSOR_SIZE = 2048


def variable_summaries(var):
    """Attaches summaries to a Tensor for TensorBoard visualization."""
    with tf.compat.v1.name_scope('summaries'):
        mean = tf.reduce_mean(input_tensor=var)
        tf.compat.v1.summary.scalar('mean', mean)
        with tf.compat.v1.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(input_tensor=tf.square(var - mean)))
        tf.compat.v1.summary.scalar('stddev', stddev)
        tf.compat.v1.summary.scalar('max', tf.reduce_max(input_tensor=var))
        tf.compat.v1.summary.scalar('min', tf.reduce_min(input_tensor=var))
        tf.compat.v1.summary.histogram('histogram', var)


def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor,
                           learning_rate):
    """Adds a new softmax and fully-connected layer for training.

    Args:
      class_count: Integer of how many categories we're trying to recognize.
      final_tensor_name: Name string for the new final node.
      bottleneck_tensor: The output of the main CNN graph.
      learning_rate: Float learning rate for gradient descent.

    Returns:
      Tensors for training step, cross entropy, bottleneck input,
      ground truth input, and final output.
    """
    with tf.compat.v1.name_scope('input'):
        bottleneck_input = tf.compat.v1.placeholder_with_default(
            bottleneck_tensor, shape=[None, BOTTLENECK_TENSOR_SIZE],
            name='BottleneckInputPlaceholder')
        ground_truth_input = tf.compat.v1.placeholder(
            tf.float32, [None, class_count], name='GroundTruthInput')

    layer_name = 'final_training_ops'
    with tf.compat.v1.name_scope(layer_name):
        with tf.compat.v1.name_scope('weights'):
            layer_weights = tf.Variable(
                tf.random.truncated_normal(
                    [BOTTLENECK_TENSOR_SIZE, class_count], stddev=0.001),
                name='final_weights')
            variable_summaries(layer_weights)
        with tf.compat.v1.name_scope('biases'):
            layer_biases = tf.Variable(
                tf.zeros([class_count]), name='final_biases')
            variable_summaries(layer_biases)
        with tf.compat.v1.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
            tf.compat.v1.summary.histogram('pre_activations', logits)

    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
    tf.compat.v1.summary.histogram('activations', final_tensor)

    with tf.compat.v1.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.stop_gradient(ground_truth_input), logits=logits)
        with tf.compat.v1.name_scope('total'):
            cross_entropy_mean = tf.reduce_mean(input_tensor=cross_entropy)
    tf.compat.v1.summary.scalar('cross_entropy', cross_entropy_mean)

    with tf.compat.v1.name_scope('train'):
        train_step = tf.compat.v1.train.GradientDescentOptimizer(
            learning_rate).minimize(cross_entropy_mean)

    return (train_step, cross_entropy_mean, bottleneck_input,
            ground_truth_input, final_tensor)


def add_evaluation_step(result_tensor, ground_truth_tensor):
    """Inserts operations to evaluate the accuracy of results.

    Args:
      result_tensor: The new final node that produces results.
      ground_truth_tensor: The node we feed ground truth data into.

    Returns:
      Tuple of (evaluation step, prediction).
    """
    with tf.compat.v1.name_scope('accuracy'):
        with tf.compat.v1.name_scope('correct_prediction'):
            prediction = tf.argmax(input=result_tensor, axis=1)
            correct_prediction = tf.equal(
                prediction, tf.argmax(input=ground_truth_tensor, axis=1))
        with tf.compat.v1.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(
                input_tensor=tf.cast(correct_prediction, tf.float32))
    tf.compat.v1.summary.scalar('accuracy', evaluation_step)
    return evaluation_step, prediction


def run_training_loop(sess, FLAGS, image_lists, do_distort_images,
                      train_step, cross_entropy, bottleneck_input,
                      ground_truth_input, evaluation_step, merged,
                      train_writer, validation_writer,
                      jpeg_data_tensor, bottleneck_tensor,
                      distorted_jpeg_data_tensor=None,
                      distorted_image_tensor=None,
                      resized_image_tensor=None):
    """Runs the full training loop for the specified number of steps.

    Args:
      sess: Active TensorFlow Session.
      FLAGS: Parsed argument flags.
      image_lists: Dictionary of training images for each label.
      do_distort_images: Boolean whether distortions are active.
      train_step, cross_entropy, bottleneck_input, ground_truth_input,
      evaluation_step, merged, train_writer, validation_writer: TF graph ops.
      jpeg_data_tensor, bottleneck_tensor: Tensors for bottleneck computation.
      distorted_jpeg_data_tensor, distorted_image_tensor,
      resized_image_tensor: Distortion tensors (only used if do_distort_images).

    Returns:
      Training time in seconds.
    """
    start_time = time.perf_counter()

    for i in range(FLAGS.how_many_training_steps):
        if do_distort_images:
            train_bottlenecks, train_ground_truth = get_random_distorted_bottlenecks(
                sess, image_lists, FLAGS.train_batch_size, 'training',
                FLAGS.image_dir, distorted_jpeg_data_tensor,
                distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
        else:
            train_bottlenecks, train_ground_truth, _ = get_random_cached_bottlenecks(
                sess, image_lists, FLAGS.train_batch_size, 'training',
                FLAGS.bottleneck_dir, FLAGS.image_dir,
                jpeg_data_tensor, bottleneck_tensor)

        train_summary, _ = sess.run(
            [merged, train_step],
            feed_dict={bottleneck_input: train_bottlenecks,
                       ground_truth_input: train_ground_truth})
        train_writer.add_summary(train_summary, i)

        is_last_step = (i + 1 == FLAGS.how_many_training_steps)
        if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
            train_accuracy, cross_entropy_value = sess.run(
                [evaluation_step, cross_entropy],
                feed_dict={bottleneck_input: train_bottlenecks,
                           ground_truth_input: train_ground_truth})
            print('%s: Step %d: Train accuracy = %.1f%%' % (
                datetime.now(), i, train_accuracy * 100))
            print('%s: Step %d: Cross entropy = %f' % (
                datetime.now(), i, cross_entropy_value))

            validation_bottlenecks, validation_ground_truth, _ = (
                get_random_cached_bottlenecks(
                    sess, image_lists, FLAGS.validation_batch_size, 'validation',
                    FLAGS.bottleneck_dir, FLAGS.image_dir,
                    jpeg_data_tensor, bottleneck_tensor))
            validation_summary, validation_accuracy = sess.run(
                [merged, evaluation_step],
                feed_dict={bottleneck_input: validation_bottlenecks,
                           ground_truth_input: validation_ground_truth})
            validation_writer.add_summary(validation_summary, i)
            print('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' % (
                datetime.now(), i, validation_accuracy * 100,
                len(validation_bottlenecks)))

    end_time = time.perf_counter()
    training_time = end_time - start_time
    print(f"Training Time: {training_time:.4f} seconds")

    with open("training_time_log.txt", "a") as f:
        f.write(f"{training_time:.4f}\n")

    return training_time
