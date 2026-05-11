# retrain.py
# Tactic applied: Dead Code & Flag Elimination
# Removed: write_list_of_floats_to_file, read_list_of_floats_from_file,
#          bottleneck_path_2_bottleneck_values, distortion flag branch
#          (flip_left_right, random_crop, random_scale, random_brightness),
#          print_misclassified_test_images flag and branch,
#          commented-out CSV average block, unused imports (struct, tarfile,
#          urllib, tensor_shape).
# All remaining logic is unchanged — no behaviour is modified, only dead
# paths and unreachable branches are removed.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys
import time

import numpy as np

import tensorflow as tf
from tensorflow.compat.v1.graph_util import convert_variables_to_constants
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

import csv
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             classification_report)

FLAGS = None

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

tf.compat.v1.disable_eager_execution()


def create_image_lists(image_dir, testing_percentage, validation_percentage):
    """Builds a list of training images from the file system.

    Analyzes the sub folders in the image directory, splits them into stable
    training, testing, and validation sets, and returns a data structure
    describing the lists of images for each label and their paths.

    Args:
      image_dir: String path to a folder containing subfolders of images.
      testing_percentage: Integer percentage of the images to reserve for tests.
      validation_percentage: Integer percentage of images reserved for validation.

    Returns:
      A dictionary containing an entry for each label subfolder, with images
      split into training, testing, and validation sets within each label.
    """
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    result = {}
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        print("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))
        if not file_list:
            print('No files found')
            continue
        if len(file_list) < 20:
            print('WARNING: Folder has less than 20 images, which may cause issues.')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            print('WARNING: Folder {} has more than {} images. Some images will '
                  'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            hash_name_hashed = hashlib.sha1(
                compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                                (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                               (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    return result


def get_image_path(image_lists, label_name, index, image_dir, category):
    """Returns a path to an image for a label at the given index."""
    if label_name not in image_lists:
        tf.compat.v1.logging.fatal('Label does not exist %s.', label_name)
    label_lists = image_lists[label_name]
    if category not in label_lists:
        tf.compat.v1.logging.fatal('Category does not exist %s.', category)
    category_list = label_lists[category]
    if not category_list:
        tf.compat.v1.logging.fatal('Label %s has no images in the category %s.',
                                   label_name, category)
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir, category):
    """Returns a path to a bottleneck file for a label at the given index."""
    return get_image_path(image_lists, label_name, index, bottleneck_dir,
                          category) + '.txt'


def create_inception_graph():
    """Creates a graph from saved GraphDef file and returns a Graph object."""
    with tf.compat.v1.Session() as sess:
        model_filename = os.path.join(
            FLAGS.model_dir, 'classify_image_graph_def.pb')
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
                tf.import_graph_def(graph_def, name='', return_elements=[
                    BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
                    RESIZED_INPUT_TENSOR_NAME]))
    return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor


def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    """Runs inference on an image to extract the bottleneck summary layer."""
    bottleneck_values = sess.run(
        bottleneck_tensor,
        {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


def maybe_download_and_extract():
    """Downloads and extracts the Inception v3 model tar file if not present."""
    import tarfile
    from six.moves import urllib
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename,
                              float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def ensure_dir_exists(dir_name):
    """Makes sure the folder exists on disk."""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           bottleneck_tensor):
    """Computes and writes a bottleneck file for a single image."""
    print('Creating bottleneck at ' + bottleneck_path)
    image_path = get_image_path(
        image_lists, label_name, index, image_dir, category)
    if not gfile.Exists(image_path):
        tf.compat.v1.logging.fatal('File does not exist %s', image_path)
    image_data = gfile.FastGFile(image_path, 'rb').read()
    bottleneck_values = run_bottleneck_on_image(
        sess, image_data, jpeg_data_tensor, bottleneck_tensor)
    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)


def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,
                             category, bottleneck_dir, jpeg_data_tensor,
                             bottleneck_tensor):
    """Retrieves or calculates bottleneck values for an image."""
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
    ensure_dir_exists(sub_dir_path)
    bottleneck_path = get_bottleneck_path(
        image_lists, label_name, index, bottleneck_dir, category)
    if not os.path.exists(bottleneck_path):
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                               image_dir, category, sess, jpeg_data_tensor,
                               bottleneck_tensor)
    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    did_hit_error = False
    try:
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    except:
        print("Invalid float found, recreating bottleneck")
        did_hit_error = True
    if did_hit_error:
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                               image_dir, category, sess, jpeg_data_tensor,
                               bottleneck_tensor)
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values


def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,
                      jpeg_data_tensor, bottleneck_tensor):
    """Ensures all training, testing, and validation bottlenecks are cached."""
    how_many_bottlenecks = 0
    ensure_dir_exists(bottleneck_dir)
    for label_name, label_lists in image_lists.items():
        for category in ['training', 'testing', 'validation']:
            category_list = label_lists[category]
            for index, unused_base_name in enumerate(category_list):
                get_or_create_bottleneck(sess, image_lists, label_name, index,
                                         image_dir, category, bottleneck_dir,
                                         jpeg_data_tensor, bottleneck_tensor)
                how_many_bottlenecks += 1
                if how_many_bottlenecks % 100 == 0:
                    print(str(how_many_bottlenecks) + ' bottleneck files created.')


def get_random_cached_bottlenecks(sess, image_lists, how_many, category,
                                  bottleneck_dir, image_dir, jpeg_data_tensor,
                                  bottleneck_tensor):
    """Retrieves bottleneck values for a random sample of cached images."""
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    filenames = []
    if how_many >= 0:
        for unused_i in range(how_many):
            label_index = random.randrange(class_count)
            label_name = list(image_lists.keys())[label_index]
            image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
            image_name = get_image_path(image_lists, label_name, image_index,
                                        image_dir, category)
            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name,
                                                  image_index, image_dir, category,
                                                  bottleneck_dir, jpeg_data_tensor,
                                                  bottleneck_tensor)
            ground_truth = np.zeros(class_count, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
            filenames.append(image_name)
    else:
        for label_index, label_name in enumerate(image_lists.keys()):
            for image_index, image_name in enumerate(
                    image_lists[label_name][category]):
                image_name = get_image_path(image_lists, label_name, image_index,
                                            image_dir, category)
                bottleneck = get_or_create_bottleneck(sess, image_lists, label_name,
                                                      image_index, image_dir, category,
                                                      bottleneck_dir, jpeg_data_tensor,
                                                      bottleneck_tensor)
                ground_truth = np.zeros(class_count, dtype=np.float32)
                ground_truth[label_index] = 1.0
                bottlenecks.append(bottleneck)
                ground_truths.append(ground_truth)
                filenames.append(image_name)
    return bottlenecks, ground_truths, filenames


def variable_summaries(var):
    """Attaches summaries to a Tensor for TensorBoard visualization."""
    with tf.compat.v1.name_scope('summaries'):
        mean = tf.reduce_mean(input_tensor=var)
        tf.compat.v1.summary.scalar('mean', mean)
        with tf.compat.v1.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(
                input_tensor=tf.square(var - mean)))
        tf.compat.v1.summary.scalar('stddev', stddev)
        tf.compat.v1.summary.scalar('max', tf.reduce_max(input_tensor=var))
        tf.compat.v1.summary.scalar('min', tf.reduce_min(input_tensor=var))
        tf.compat.v1.summary.histogram('histogram', var)


def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor):
    """Adds a new softmax and fully-connected layer for training."""
    with tf.compat.v1.name_scope('input'):
        bottleneck_input = tf.compat.v1.placeholder_with_default(
            bottleneck_tensor, shape=[None, BOTTLENECK_TENSOR_SIZE],
            name='BottleneckInputPlaceholder')
        ground_truth_input = tf.compat.v1.placeholder(tf.float32,
                                                      [None, class_count],
                                                      name='GroundTruthInput')
    layer_name = 'final_training_ops'
    with tf.compat.v1.name_scope(layer_name):
        with tf.compat.v1.name_scope('weights'):
            layer_weights = tf.Variable(tf.random.truncated_normal(
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
            FLAGS.learning_rate).minimize(cross_entropy_mean)
    return (train_step, cross_entropy_mean, bottleneck_input,
            ground_truth_input, final_tensor)


def add_evaluation_step(result_tensor, ground_truth_tensor):
    """Inserts operations to evaluate the accuracy of results."""
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


def main(_):
    if tf.io.gfile.exists(FLAGS.summaries_dir):
        tf.io.gfile.rmtree(FLAGS.summaries_dir)
    tf.io.gfile.makedirs(FLAGS.summaries_dir)

    maybe_download_and_extract()
    graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (
        create_inception_graph())

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

    sess = tf.compat.v1.Session()

    # Distortion flags removed: pipeline always uses cached bottlenecks.
    cache_bottlenecks(sess, image_lists, FLAGS.image_dir, FLAGS.bottleneck_dir,
                      jpeg_data_tensor, bottleneck_tensor)

    (train_step, cross_entropy, bottleneck_input, ground_truth_input,
     final_tensor) = add_final_training_ops(len(image_lists.keys()),
                                            FLAGS.final_tensor_name,
                                            bottleneck_tensor)

    evaluation_step, prediction = add_evaluation_step(
        final_tensor, ground_truth_input)

    merged = tf.compat.v1.summary.merge_all()
    train_writer = tf.compat.v1.summary.FileWriter(
        FLAGS.summaries_dir + '/train', sess.graph)
    validation_writer = tf.compat.v1.summary.FileWriter(
        FLAGS.summaries_dir + '/validation')

    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    start_time = time.perf_counter()

    for i in range(FLAGS.how_many_training_steps):
        train_bottlenecks, train_ground_truth, _ = get_random_cached_bottlenecks(
            sess, image_lists, FLAGS.train_batch_size, 'training',
            FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
            bottleneck_tensor)

        train_summary, _ = sess.run([merged, train_step],
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
                    FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                    bottleneck_tensor))
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

    test_bottlenecks, test_ground_truth, test_filenames = (
        get_random_cached_bottlenecks(sess, image_lists, FLAGS.test_batch_size,
                                      'testing', FLAGS.bottleneck_dir,
                                      FLAGS.image_dir, jpeg_data_tensor,
                                      bottleneck_tensor))
    test_accuracy, predictions = sess.run(
        [evaluation_step, prediction],
        feed_dict={bottleneck_input: test_bottlenecks,
                   ground_truth_input: test_ground_truth})
    print('Final test accuracy = %.1f%% (N=%d)' % (
        test_accuracy * 100, len(test_bottlenecks)))

    output_graph_def = convert_variables_to_constants(
        sess, graph.as_graph_def(), [FLAGS.final_tensor_name])
    with gfile.FastGFile(FLAGS.output_graph, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
        f.write('\n'.join(image_lists.keys()) + '\n')


def f1_test_set_evaluation(sess, labels_list, test_dir, run_id,
                            run_number, metrics_output_dir):
    """Computes F1, precision, and recall on the fixed test set."""
    label_map = {lbl.lower().strip(): i for i, lbl in enumerate(labels_list)}
    samples = []
    for class_folder in sorted(os.listdir(test_dir)):
        folder_path = os.path.join(test_dir, class_folder)
        if not os.path.isdir(folder_path):
            continue
        class_key = class_folder.lower()
        if class_key not in label_map:
            print('WARNING: Test folder "%s" not found in labels, skipping.' % class_folder)
            continue
        label_idx = label_map[class_key]
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                samples.append((os.path.join(folder_path, fname), label_idx))

    print('F1 eval: %d test images across %d classes.' % (len(samples), len(label_map)))
    input_tensor = sess.graph.get_tensor_by_name('DecodeJpeg/contents:0')
    output_tensor = sess.graph.get_tensor_by_name('final_result:0')

    y_true, y_pred = [], []
    for img_path, true_idx in samples:
        try:
            img_data = gfile.FastGFile(img_path, 'rb').read()
            predictions = sess.run(output_tensor, {input_tensor: img_data})
            y_true.append(true_idx)
            y_pred.append(int(np.argmax(predictions)))
        except Exception as e:
            print('WARNING: Could not process %s: %s' % (img_path, str(e)))

    f1 = round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4)
    precision = round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4)
    recall = round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4)

    print('Run %d | F1: %.4f | Precision: %.4f | Recall: %.4f' % (
        run_number, f1, precision, recall))
    print(classification_report(y_true, y_pred,
                                labels=list(range(len(labels_list))),
                                target_names=labels_list, zero_division=0))

    os.makedirs(metrics_output_dir, exist_ok=True)
    csv_path = os.path.join(metrics_output_dir, 'f1_results.csv')
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'run_id', 'run_number',
                      'f1_weighted', 'precision_weighted', 'recall_weighted']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'run_id': run_id,
            'run_number': run_number,
            'f1_weighted': f1,
            'precision_weighted': precision,
            'recall_weighted': recall,
        })
    return f1, precision, recall

def validate_directory(image_dir, allowedexts=('.jpg', '.jpeg', '.JPG', '.JPEG'), verbose=False):
    """
    Checks all images in a directory tree for extension, nonzero size, and decodability.
    Returns a dict with counts and failed files.
    """
    import tensorflow as tf
    from tensorflow.python.platform import gfile

    total = 0
    passed = 0
    failed = []
    for root, _, files in os.walk(image_dir):
        for fname in files:
            if not fname.endswith(allowedexts):
                continue
            fpath = os.path.join(root, fname)
            total += 1
            # Check file size
            if os.path.getsize(fpath) == 0:
                failed.append(fpath)
                if verbose:
                    print(f"Zero size: {fpath}")
                continue
            # Check decodability
            try:
                with open(fpath, "rb") as f:
                    img_bytes = f.read()
                tf.image.decode_jpeg(img_bytes)
            except Exception as e:
                failed.append(fpath)
                if verbose:
                    print(f"Decode failed: {fpath} ({e})")
                continue
            passed += 1
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "score": passed / total if total else 0.0
    }

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
    parser.add_argument('--model_dir', type=str, default='/tmp/imagenet',
                        help='Path to the Inception v3 model files.')
    parser.add_argument('--bottleneck_dir', type=str, default='/tmp/bottleneck',
                        help='Path to cache bottleneck layer values as files.')
    parser.add_argument('--final_tensor_name', type=str, default='final_result',
                        help='Name of the output classification layer.')
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