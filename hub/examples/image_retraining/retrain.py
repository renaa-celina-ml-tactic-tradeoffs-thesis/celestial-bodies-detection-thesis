# retrain.py
# ML Lifecycle Stage: Orchestration
# This file is the entry point and pipeline orchestrator only.
# It wires together the discrete lifecycle modules:
#   data_loader    -> image discovery and splitting
#   bottleneck_cache -> feature extraction and caching
#   augmentation   -> data preprocessing / distortion
#   model_trainer  -> training loop and ops
#   evaluator      -> F1 scoring and reliability
#   exporter       -> graph export and model download
#
# No pipeline logic lives here. Each stage is delegated to its module.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

# --- Lifecycle module imports ---
from data_loader import create_image_lists
from bottleneck_cache import cache_bottlenecks, get_random_cached_bottlenecks
from augmentation import (should_distort_images, add_input_distortions,
                          get_random_distorted_bottlenecks)
from model_trainer import (add_final_training_ops, add_evaluation_step,
                           run_training_loop)
from evaluator import f1_test_set_evaluation, validate_directory
from exporter import maybe_download_and_extract, create_inception_graph, export_model

FLAGS = None


def main(_):
    # Setup TensorBoard summaries directory
    if tf.io.gfile.exists(FLAGS.summaries_dir):
        tf.io.gfile.rmtree(FLAGS.summaries_dir)
    tf.io.gfile.makedirs(FLAGS.summaries_dir)

    # --- Exporter: download pre-trained model ---
    maybe_download_and_extract(FLAGS.model_dir)
    graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (
        create_inception_graph(FLAGS.model_dir))

    # --- Data Loader: discover and split images ---
    image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,
                                     FLAGS.validation_percentage)
    class_count = len(image_lists.keys())
    if class_count == 0:
        print('No valid folders of images found at ' + FLAGS.image_dir)
        return -1
    if class_count == 1:
        print('Only one valid folder of images found at ' + FLAGS.image_dir +
              ' - multiple classes are needed for classification.')
        return -1

    # --- Augmentation: check if distortions are requested ---
    do_distort_images = should_distort_images(
        FLAGS.flip_left_right, FLAGS.random_crop,
        FLAGS.random_scale, FLAGS.random_brightness)

    sess = tf.compat.v1.Session()

    distorted_jpeg_data_tensor = None
    distorted_image_tensor = None

    if do_distort_images:
        # --- Augmentation: build distortion graph ---
        distorted_jpeg_data_tensor, distorted_image_tensor = add_input_distortions(
            FLAGS.flip_left_right, FLAGS.random_crop,
            FLAGS.random_scale, FLAGS.random_brightness)
    else:
        # --- Bottleneck Cache: pre-compute and cache all bottlenecks ---
        cache_bottlenecks(sess, image_lists, FLAGS.image_dir,
                          FLAGS.bottleneck_dir, jpeg_data_tensor, bottleneck_tensor)

    # --- Model Trainer: build final classification layer ---
    (train_step, cross_entropy, bottleneck_input, ground_truth_input,
     final_tensor) = add_final_training_ops(
        class_count, FLAGS.final_tensor_name, bottleneck_tensor, FLAGS.learning_rate)

    evaluation_step, prediction = add_evaluation_step(final_tensor, ground_truth_input)

    merged = tf.compat.v1.summary.merge_all()
    train_writer = tf.compat.v1.summary.FileWriter(
        FLAGS.summaries_dir + '/train', sess.graph)
    validation_writer = tf.compat.v1.summary.FileWriter(
        FLAGS.summaries_dir + '/validation')

    sess.run(tf.compat.v1.global_variables_initializer())

    # --- Model Trainer: run training loop ---
    run_training_loop(
        sess, FLAGS, image_lists, do_distort_images,
        train_step, cross_entropy, bottleneck_input, ground_truth_input,
        evaluation_step, merged, train_writer, validation_writer,
        jpeg_data_tensor, bottleneck_tensor,
        distorted_jpeg_data_tensor, distorted_image_tensor, resized_image_tensor)

    # --- Evaluator: run final test evaluation ---
    test_bottlenecks, test_ground_truth, test_filenames = (
        get_random_cached_bottlenecks(
            sess, image_lists, FLAGS.test_batch_size, 'testing',
            FLAGS.bottleneck_dir, FLAGS.image_dir,
            jpeg_data_tensor, bottleneck_tensor))
    test_accuracy, predictions = sess.run(
        [evaluation_step, prediction],
        feed_dict={bottleneck_input: test_bottlenecks,
                   ground_truth_input: test_ground_truth})
    print('Final test accuracy = %.1f%% (N=%d)' % (
        test_accuracy * 100, len(test_bottlenecks)))

    if FLAGS.print_misclassified_test_images:
        print('=== MISCLASSIFIED TEST IMAGES ===')
        for i, test_filename in enumerate(test_filenames):
            if predictions[i] != test_ground_truth[i].argmax():
                print('%70s  %s' % (test_filename,
                                    list(image_lists.keys())[predictions[i]]))

    # --- Exporter: save frozen graph and labels ---
    export_model(sess, graph, FLAGS.final_tensor_name,
                 FLAGS.output_graph, FLAGS.output_labels, image_lists)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='',
                        help='Path to folders of labeled images.')
    parser.add_argument('--output_graph', type=str, default='/tmp/output_graph.pb',
                        help='Where to save the trained graph.')
    parser.add_argument('--output_labels', type=str, default='/tmp/output_labels.txt',
                        help='Where to save the trained graph labels.')
    parser.add_argument('--summaries_dir', type=str, default='/tmp/retrain_logs',
                        help='Where to save summary logs for TensorBoard.')
    parser.add_argument('--how_many_training_steps', type=int, default=4000,
                        help='How many training steps to run before ending.')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='How large a learning rate to use when training.')
    parser.add_argument('--testing_percentage', type=int, default=10,
                        help='What percentage of images to use as a test set.')
    parser.add_argument('--validation_percentage', type=int, default=10,
                        help='What percentage of images to use as a validation set.')
    parser.add_argument('--eval_step_interval', type=int, default=10,
                        help='How often to evaluate the training results.')
    parser.add_argument('--train_batch_size', type=int, default=100,
                        help='How many images to train on at a time.')
    parser.add_argument('--test_batch_size', type=int, default=-1,
                        help='How many images to test on. -1 uses the full test set.')
    parser.add_argument('--validation_batch_size', type=int, default=100,
                        help='How many images to use in an evaluation batch.')
    parser.add_argument('--print_misclassified_test_images', default=False,
                        action='store_true',
                        help='Whether to print misclassified test images.')
    parser.add_argument('--model_dir', type=str, default='/tmp/imagenet',
                        help='Path to the Inception v3 model files.')
    parser.add_argument('--bottleneck_dir', type=str, default='/tmp/bottleneck',
                        help='Path to cache bottleneck layer values as files.')
    parser.add_argument('--final_tensor_name', type=str, default='final_result',
                        help='Name of the output classification layer.')
    parser.add_argument('--flip_left_right', default=False, action='store_true',
                        help='Whether to randomly flip training images horizontally.')
    parser.add_argument('--random_crop', type=int, default=0,
                        help='Percentage margin to randomly crop off training images.')
    parser.add_argument('--random_scale', type=int, default=0,
                        help='Percentage to randomly scale up training images by.')
    parser.add_argument('--random_brightness', type=int, default=0,
                        help='Percentage to randomly multiply training image pixels by.')
    parser.add_argument('--test_dir', type=str, default='/test_data',
                        help='Directory with class-subfolder test images for F1 evaluation.')
    parser.add_argument('--run_id', type=str, default='baseline',
                        help='Identifier for this measurement state.')
    parser.add_argument('--eval_runs', type=int, default=5,
                        help='Number of times to retrain and evaluate for F1 averaging.')
    parser.add_argument('--metrics_output_dir', type=str, default='../../../measurements',
                        help='Directory to write F1 CSV results.')

    FLAGS, unparsed = parser.parse_known_args()

    # # Reset the f1 results CSV at the start of each run
    # os.makedirs(FLAGS.metrics_output_dir, exist_ok=True)
    # csv_path = os.path.join(FLAGS.metrics_output_dir, 'f1_results.csv')
    # if os.path.isfile(csv_path):
    #     os.remove(csv_path)


    # Run multiple training + evaluation cycles to get an average F1 score, since it can vary from run to run.
    all_f1, all_precision, all_recall = [], [], []

    for run_num in range(1, FLAGS.eval_runs + 1):
        print('\n=== Training + Eval Run %d/%d (run_id: %s) ===' % (
            run_num, FLAGS.eval_runs, FLAGS.run_id))
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.disable_eager_execution()
        main([sys.argv[0]] + unparsed)

        # Load the exported graph for F1 evaluation
        eval_graph = tf.compat.v1.Graph()
        with eval_graph.as_default():
            graph_def = tf.compat.v1.GraphDef()
            with gfile.FastGFile(FLAGS.output_graph, 'rb') as f:
                graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

        labels_list = [l.strip() for l in open(FLAGS.output_labels).readlines()]

        with tf.compat.v1.Session(graph=eval_graph) as eval_sess:
            f1, precision, recall = f1_test_set_evaluation(
                eval_sess, labels_list, FLAGS.test_dir,
                FLAGS.run_id, run_num, FLAGS.metrics_output_dir)
            all_f1.append(f1)
            all_precision.append(precision)
            all_recall.append(recall)
            
    avg_f1 = round(float(np.mean(all_f1)), 4)
    avg_precision = round(float(np.mean(all_precision)), 4)
    avg_recall = round(float(np.mean(all_recall)), 4)


    ### ORIGINAL AVERAGE PRINTING AND CSV LOGGING FOR F1 SCORE - COMMENTED OUT TO PREVENT DUPLICATE LOGGING DURING MULTIPLE RUNS, BUT CAN BE RE-ENABLED IF DESIRED. ###
    # print('\n=== AVERAGE over %d runs | F1: %.4f | Precision: %.4f | Recall: %.4f ===' % (
    #     FLAGS.eval_runs, avg_f1, avg_precision, avg_recall))

    # csv_path = os.path.join(FLAGS.metrics_output_dir, 'f1_results.csv')
    # with open(csv_path, 'a', newline='') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=[
    #         'timestamp', 'run_id', 'run_number',
    #         'f1_weighted', 'precision_weighted', 'recall_weighted'])
    #     writer.writerow({
    #         'timestamp': datetime.now().isoformat(timespec='seconds'),
    #         'run_id': FLAGS.run_id + '_AVG',
    #         'run_number': 0,
    #         'f1_weighted': avg_f1,
    #         'precision_weighted': avg_precision,
    #         'recall_weighted': avg_recall,
    #     })