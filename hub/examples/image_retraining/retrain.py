# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Simple transfer learning with an Inception v3 architecture model which
displays summaries in TensorBoard.

This example shows how to take a Inception v3 architecture model trained on
ImageNet images, and train a new top layer that can recognize other classes of
images.

The top layer receives as input a 2048-dimensional vector for each image. We
train a softmax layer on top of this representation. Assuming the softmax layer
contains N labels, this corresponds to learning N + 2048*N model parameters
corresponding to the learned biases and weights.

Here's an example, which assumes you have a folder containing class-named
subfolders, each full of images for each label. The example folder flower_photos
should have a structure like this:

~/flower_photos/daisy/photo1.jpg
~/flower_photos/daisy/photo2.jpg
...
~/flower_photos/rose/anotherphoto77.jpg
...
~/flower_photos/sunflower/somepicture.jpg

The subfolder names are important, since they define what label is applied to
each image, but the filenames themselves don't matter. Once your images are
prepared, you can run the training with a command like this:

bazel build tensorflow/examples/image_retraining:retrain && \
bazel-bin/tensorflow/examples/image_retraining/retrain \
--image_dir ~/flower_photos

You can replace the image_dir argument with any folder containing subfolders of
images. The label for each image is taken from the name of the subfolder it's
in.

This produces a new model file that can be loaded and run by any TensorFlow
program, for example the label_image sample code.


To use with TensorBoard:

By default, this script will log summaries to /tmp/retrain_logs directory

Visualize the summaries with this command:

tensorboard --logdir /tmp/retrain_logs

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import hashlib
import os.path
import random
import re
import struct
import sys
import tarfile
import time

import numpy as np
from six.moves import urllib

import tensorflow as tf
from tensorflow.compat.v1.graph_util import convert_variables_to_constants
# from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

# Libraries for system accuracy evaluation
import csv
from sklearn.metrics import (f1_score, precision_score, recall_score, classification_report)


FLAGS = None

# These are all parameters that are tied to the particular model architecture
# we're using for Inception v3. These include things like tensor names and their
# sizes. If you want to adapt this script to work with another model, you will
# need to update these to reflect the values in the network you're using.
# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long
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
      A dictionary containing an entry for each label subfolder, with images split
      into training, testing, and validation sets within each label.
    """
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    result = {}
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
    # The root directory comes first, so skip it.
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
            # We want to ignore anything after '_nohash_' in the file name when
            # deciding which set to put an image in, the data set creator has a way of
            # grouping photos that are close variations of each other. For example
            # this is used in the plant disease data set to group multiple pictures of
            # the same leaf.
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            # This looks a bit magical, but we need to decide whether this file should
            # go into the training, testing, or validation sets, and we want to keep
            # existing files in the same set even if more files are subsequently
            # added.
            # To do that, we need a stable way of deciding based on just the file name
            # itself, so we do a hash of that and then use that to generate a
            # probability value that we use to assign it.
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
    """"Returns a path to an image for a label at the given index.

    Args:
      image_lists: Dictionary of training images for each label.
      label_name: Label string we want to get an image for.
      index: Int offset of the image we want. This will be moduloed by the
      available number of images for the label, so it can be arbitrarily large.
      image_dir: Root folder string of the subfolders containing the training
      images.
      category: Name string of set to pull images from - training, testing, or
      validation.

    Returns:
      File system path string to an image that meets the requested parameters.

    """
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


def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir,
                        category):
    """"Returns a path to a bottleneck file for a label at the given index.

    Args:
      image_lists: Dictionary of training images for each label.
      label_name: Label string we want to get an image for.
      index: Integer offset of the image we want. This will be moduloed by the
      available number of images for the label, so it can be arbitrarily large.
      bottleneck_dir: Folder string holding cached files of bottleneck values.
      category: Name string of set to pull images from - training, testing, or
      validation.

    Returns:
      File system path string to an image that meets the requested parameters.
    """
    return get_image_path(image_lists, label_name, index, bottleneck_dir,
                          category) + '.txt'


def create_inception_graph():
    """"Creates a graph from saved GraphDef file and returns a Graph object.

    Returns:
      Graph holding the trained Inception network, and various tensors we'll be
      manipulating.
    """
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


def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            bottleneck_tensor):
    """Runs inference on an image to extract the 'bottleneck' summary layer.

    Args:
      sess: Current active TensorFlow Session.
      image_data: String of raw JPEG data.
      image_data_tensor: Input data layer in the graph.
      bottleneck_tensor: Layer before the final softmax.

    Returns:
      Numpy array of bottleneck values.
    """
    bottleneck_values = sess.run(
        bottleneck_tensor,
        {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


def maybe_download_and_extract():
    """Download and extract model tar file.

    If the pretrained model we're using doesn't already exist, this function
    downloads it from the TensorFlow.org website and unpacks it into a directory.
    """
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

        filepath, _ = urllib.request.urlretrieve(DATA_URL,
                                                 filepath,
                                                 _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def ensure_dir_exists(dir_name):
    """Makes sure the folder exists on disk.

    Args:
      dir_name: Path string to the folder we want to create.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def write_list_of_floats_to_file(list_of_floats, file_path):
    """Writes a given list of floats to a binary file.

    Args:
      list_of_floats: List of floats we want to write to a file.
      file_path: Path to a file where list of floats will be stored.

    """

    s = struct.pack('d' * BOTTLENECK_TENSOR_SIZE, *list_of_floats)
    with open(file_path, 'wb') as f:
        f.write(s)


def read_list_of_floats_from_file(file_path):
    """Reads list of floats from a given file.

    Args:
      file_path: Path to a file where list of floats was stored.
    Returns:
      Array of bottleneck values (list of floats).

    """

    with open(file_path, 'rb') as f:
        s = struct.unpack('d' * BOTTLENECK_TENSOR_SIZE, f.read())
        return list(s)


bottleneck_path_2_bottleneck_values = {}


def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor, bottleneck_tensor):
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
    """Retrieves or calculates bottleneck values for an image.

    If a cached version of the bottleneck data exists on-disk, return that,
    otherwise calculate the data and save it to disk for future use.

    Args:
      sess: The current active TensorFlow Session.
      image_lists: Dictionary of training images for each label.
      label_name: Label string we want to get an image for.
      index: Integer offset of the image we want. This will be modulo-ed by the
      available number of images for the label, so it can be arbitrarily large.
      image_dir: Root folder string  of the subfolders containing the training
      images.
      category: Name string of which  set to pull images from - training, testing,
      or validation.
      bottleneck_dir: Folder string holding cached files of bottleneck values.
      jpeg_data_tensor: The tensor to feed loaded jpeg data into.
      bottleneck_tensor: The output tensor for the bottleneck values.

    Returns:
      Numpy array of values produced by the bottleneck layer for the image.
    """
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
    ensure_dir_exists(sub_dir_path)
    bottleneck_path = get_bottleneck_path(
        image_lists, label_name, index, bottleneck_dir, category)
    if not os.path.exists(bottleneck_path):
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                               image_dir, category, sess, jpeg_data_tensor, bottleneck_tensor)
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
                               image_dir, category, sess, jpeg_data_tensor, bottleneck_tensor)
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        # Allow exceptions to propagate here, since they shouldn't happen after a fresh creation
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values


def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,
                      jpeg_data_tensor, bottleneck_tensor):
    """Ensures all the training, testing, and validation bottlenecks are cached.

    Because we're likely to read the same image multiple times (if there are no
    distortions applied during training) it can speed things up a lot if we
    calculate the bottleneck layer values once for each image during
    preprocessing, and then just read those cached values repeatedly during
    training. Here we go through all the images we've found, calculate those
    values, and save them off.

    Args:
      sess: The current active TensorFlow Session.
      image_lists: Dictionary of training images for each label.
      image_dir: Root folder string of the subfolders containing the training
      images.
      bottleneck_dir: Folder string holding cached files of bottleneck values.
      jpeg_data_tensor: Input tensor for jpeg data from file.
      bottleneck_tensor: The penultimate output layer of the graph.

    Returns:
      Nothing.
    """
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
                    print(str(how_many_bottlenecks) +
                          ' bottleneck files created.')


def get_random_cached_bottlenecks(sess, image_lists, how_many, category,
                                  bottleneck_dir, image_dir, jpeg_data_tensor,
                                  bottleneck_tensor):
    """Retrieves bottleneck values for cached images.

    If no distortions are being applied, this function can retrieve the cached
    bottleneck values directly from disk for images. It picks a random set of
    images from the specified category.

    Args:
      sess: Current TensorFlow Session.
      image_lists: Dictionary of training images for each label.
      how_many: If positive, a random sample of this size will be chosen.
      If negative, all bottlenecks will be retrieved.
      category: Name string of which set to pull from - training, testing, or
      validation.
      bottleneck_dir: Folder string holding cached files of bottleneck values.
      image_dir: Root folder string of the subfolders containing the training
      images.
      jpeg_data_tensor: The layer to feed jpeg image data into.
      bottleneck_tensor: The bottleneck output layer of the CNN graph.

    Returns:
      List of bottleneck arrays, their corresponding ground truths, and the
      relevant filenames.
    """
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    filenames = []
    if how_many >= 0:
        # Retrieve a random sample of bottlenecks.
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
        # Retrieve all bottlenecks.
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


def get_random_distorted_bottlenecks(
        sess, image_lists, how_many, category, image_dir, input_jpeg_tensor,
        distorted_image, resized_input_tensor, bottleneck_tensor):
    """Retrieves bottleneck values for training images, after distortions.

    Samples from a pre-generated in-memory pool (populated by
    build_distorted_bottleneck_pool) when available, falling back to on-the-fly
    computation otherwise.

    Args:
      sess: Current TensorFlow Session.
      image_lists: Dictionary of training images for each label.
      how_many: The integer number of bottleneck values to return.
      category: Name string of which set of images to fetch - training, testing,
      or validation.
      image_dir: Root folder string of the subfolders containing the training
      images.
      input_jpeg_tensor: The input layer we feed the image data to.
      distorted_image: The output node of the distortion graph.
      resized_input_tensor: The input node of the recognition graph.
      bottleneck_tensor: The bottleneck output layer of the CNN graph.

    Returns:
      List of bottleneck arrays and their corresponding ground truths.
    """
    # Fast path: sample from pre-generated pool if it exists
    pool = getattr(get_random_distorted_bottlenecks, '_pool', None)
    if pool is not None and len(pool['bottlenecks']) > 0:
        pool_size = len(pool['bottlenecks'])
        indices = [random.randrange(pool_size) for _ in range(how_many)]
        bottlenecks = [pool['bottlenecks'][i] for i in indices]
        ground_truths = [pool['ground_truths'][i] for i in indices]
        return bottlenecks, ground_truths

    # Slow path: on-the-fly computation (used only when pool is not built)
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    label_names = list(image_lists.keys())
    jpeg_batch = []
    label_indices_batch = []

    for unused_i in range(how_many):
        label_index = random.randrange(class_count)
        label_name = label_names[label_index]
        image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
        image_path = get_image_path(image_lists, label_name, image_index, image_dir,
                                    category)
        if not gfile.Exists(image_path):
            tf.compat.v1.logging.fatal('File does not exist %s', image_path)
        jpeg_batch.append(gfile.FastGFile(image_path, 'rb').read())
        label_indices_batch.append(label_index)

    for jpeg_data, label_index in zip(jpeg_batch, label_indices_batch):
        distorted_image_data = sess.run(distorted_image,
                                        {input_jpeg_tensor: jpeg_data})
        bottleneck = run_bottleneck_on_image(sess, distorted_image_data,
                                             resized_input_tensor,
                                             bottleneck_tensor)
        ground_truth = np.zeros(class_count, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


def build_distorted_bottleneck_pool(
        sess, image_lists, category, image_dir, input_jpeg_tensor,
        distorted_image, resized_input_tensor, bottleneck_tensor,
        augmentations_per_image=3):
    """Pre-generates distorted bottlenecks for all training images into memory.

    Running the full Inception network on-the-fly every training step is the
    primary cause of slow training when distortions are enabled. This function
    amortises that cost by running each image through the distortion + Inception
    pipeline a fixed number of times up-front and storing the resulting 2048-d
    bottleneck vectors in RAM. Subsequent calls to get_random_distorted_bottlenecks
    simply sample from this pool, reducing per-step cost to a single NumPy lookup.

    Args:
      sess: Current TensorFlow Session.
      image_lists: Dictionary of training images for each label.
      category: Name string of which set to use ('training').
      image_dir: Root folder containing class sub-folders.
      input_jpeg_tensor: Distortion graph JPEG input placeholder.
      distorted_image: Distortion graph output tensor.
      resized_input_tensor: Inception graph image input tensor.
      bottleneck_tensor: Inception bottleneck output tensor.
      augmentations_per_image: How many distorted variants to generate per image.
        Higher values give more variety at the cost of RAM and build time.

    Returns:
      Nothing. Attaches the pool directly to get_random_distorted_bottlenecks
      as a function attribute so it is reused transparently.
    """
    class_count = len(image_lists.keys())
    label_names = list(image_lists.keys())
    all_bottlenecks = []
    all_ground_truths = []

    total_images = sum(len(image_lists[ln][category]) for ln in label_names)
    total_to_generate = total_images * augmentations_per_image
    generated = 0

    print('Building distorted bottleneck pool: %d images × %d augmentations = %d total ...' % (
        total_images, augmentations_per_image, total_to_generate))

    for label_index, label_name in enumerate(label_names):
        image_list = image_lists[label_name][category]
        for image_index in range(len(image_list)):
            image_path = get_image_path(
                image_lists, label_name, image_index, image_dir, category)
            if not gfile.Exists(image_path):
                print('WARNING: File does not exist %s, skipping.' % image_path)
                continue
            try:
                jpeg_data = gfile.FastGFile(image_path, 'rb').read()
            except Exception as e:
                print('WARNING: Could not read %s: %s' % (image_path, e))
                continue

            for _ in range(augmentations_per_image):
                try:
                    distorted_image_data = sess.run(
                        distorted_image, {input_jpeg_tensor: jpeg_data})
                    bottleneck = run_bottleneck_on_image(
                        sess, distorted_image_data,
                        resized_input_tensor, bottleneck_tensor)
                    ground_truth = np.zeros(class_count, dtype=np.float32)
                    ground_truth[label_index] = 1.0
                    all_bottlenecks.append(bottleneck)
                    all_ground_truths.append(ground_truth)
                except Exception as e:
                    print('WARNING: Augmentation failed for %s: %s' % (image_path, e))

            generated += augmentations_per_image
            if generated % 500 == 0 or generated == total_to_generate:
                print('  Pool progress: %d / %d bottlenecks generated.' % (
                    generated, total_to_generate))

    # Shuffle so training batches are well-mixed across classes
    combined = list(zip(all_bottlenecks, all_ground_truths))
    random.shuffle(combined)
    all_bottlenecks, all_ground_truths = zip(*combined) if combined else ([], [])

    get_random_distorted_bottlenecks._pool = {
        'bottlenecks': list(all_bottlenecks),
        'ground_truths': list(all_ground_truths),
    }
    print('Distorted bottleneck pool ready: %d entries.' % len(all_bottlenecks))


def should_distort_images(flip_left_right, random_crop, random_scale,
                          random_brightness):
    """Whether any distortions are enabled, from the input flags.

    Args:
      flip_left_right: Boolean whether to randomly mirror images horizontally.
      random_crop: Integer percentage setting the total margin used around the
      crop box.
      random_scale: Integer percentage of how much to vary the scale by.
      random_brightness: Integer range to randomly multiply the pixel values by.

    Returns:
      Boolean value indicating whether any distortions should be applied.
    """
    return (flip_left_right or (random_crop != 0) or (random_scale != 0) or
            (random_brightness != 0))


def add_input_distortions(flip_left_right, random_crop, random_scale,
                          random_brightness):
    """Creates the operations to apply the specified distortions.

    During training it can help to improve the results if we run the images
    through simple distortions like crops, scales, and flips. These reflect the
    kind of variations we expect in the real world, and so can help train the
    model to cope with natural data more effectively. Here we take the supplied
    parameters and construct a network of operations to apply them to an image.

    Cropping
    ~~~~~~~~

    Cropping is done by placing a bounding box at a random position in the full
    image. The cropping parameter controls the size of that box relative to the
    input image. If it's zero, then the box is the same size as the input and no
    cropping is performed. If the value is 50%, then the crop box will be half the
    width and height of the input. In a diagram it looks like this:

    <       width         >
    +---------------------+
    |                     |
    |   width - crop%     |
    |    <      >         |
    |    +------+         |
    |    |      |         |
    |    |      |         |
    |    |      |         |
    |    +------+         |
    |                     |
    |                     |
    +---------------------+

    Scaling
    ~~~~~~~

    Scaling is a lot like cropping, except that the bounding box is always
    centered and its size varies randomly within the given range. For example if
    the scale percentage is zero, then the bounding box is the same size as the
    input and no scaling is applied. If it's 50%, then the bounding box will be in
    a random range between half the width and height and full size.

    Args:
      flip_left_right: Boolean whether to randomly mirror images horizontally.
      random_crop: Integer percentage setting the total margin used around the
      crop box.
      random_scale: Integer percentage of how much to vary the scale by.
      random_brightness: Integer range to randomly multiply the pixel values by.
      graph.

    Returns:
      The jpeg input layer and the distorted result tensor.
    """

    jpeg_data = tf.compat.v1.placeholder(tf.string, name='DistortJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=MODEL_INPUT_DEPTH)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    margin_scale = 1.0 + (random_crop / 100.0)
    resize_scale = 1.0 + (random_scale / 100.0)
    margin_scale_value = tf.constant(margin_scale)
    resize_scale_value = tf.random.uniform(shape=[],
                                       minval=1.0,
                                       maxval=resize_scale)
    scale_value = tf.multiply(margin_scale_value, resize_scale_value)
    precrop_width = tf.multiply(scale_value, MODEL_INPUT_WIDTH)
    precrop_height = tf.multiply(scale_value, MODEL_INPUT_HEIGHT)
    precrop_shape = tf.stack([precrop_height, precrop_width])
    precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
    precropped_image = tf.image.resize(decoded_image_4d,
                                       precrop_shape_as_int, method=tf.image.ResizeMethod.BILINEAR)
    precropped_image_3d = tf.squeeze(precropped_image, axis=[0])
    cropped_image = tf.image.random_crop(precropped_image_3d,
                                         [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH,
                                          MODEL_INPUT_DEPTH])
    if flip_left_right:
        flipped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        flipped_image = cropped_image
    brightness_min = 1.0 - (random_brightness / 100.0)
    brightness_max = 1.0 + (random_brightness / 100.0)
    brightness_value = tf.random.uniform(shape=[],
                                     minval=brightness_min,
                                     maxval=brightness_max)
    brightened_image = tf.multiply(flipped_image, brightness_value)
    distort_result = tf.expand_dims(brightened_image, 0, name='DistortResult')
    return jpeg_data, distort_result


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
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
    """Adds a new softmax and fully-connected layer for training.

    We need to retrain the top layer to identify our new classes, so this function
    adds the right operations to the graph, along with some variables to hold the
    weights, and then sets up all the gradients for the backward pass.

    The set up for the softmax and fully-connected layers is based on:
    https://tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

    Args:
      class_count: Integer of how many categories of things we're trying to
      recognize.
      final_tensor_name: Name string for the new final node that produces results.
      bottleneck_tensor: The output of the main CNN graph.

    Returns:
      The tensors for the training and cross entropy results, and tensors for the
      bottleneck input and ground truth input.
    """
    with tf.compat.v1.name_scope('input'):
        bottleneck_input = tf.compat.v1.placeholder_with_default(
            bottleneck_tensor, shape=[None, BOTTLENECK_TENSOR_SIZE],
            name='BottleneckInputPlaceholder')

        ground_truth_input = tf.compat.v1.placeholder(tf.float32,
                                                      [None, class_count],
                                                      name='GroundTruthInput')

    # Organizing the following ops as `final_training_ops` so they're easier
    # to see in TensorBoard
    layer_name = 'final_training_ops'
    with tf.compat.v1.name_scope(layer_name):
        with tf.compat.v1.name_scope('weights'):
            layer_weights = tf.Variable(tf.random.truncated_normal(
                [BOTTLENECK_TENSOR_SIZE, class_count], stddev=0.001), name='final_weights')
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
        # Adam converges significantly faster than vanilla SGD, allowing the
        # same validation accuracy to be reached in fewer training steps.
        train_step = tf.compat.v1.train.AdamOptimizer(FLAGS.learning_rate).minimize(
            cross_entropy_mean)

    return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
            final_tensor)


def add_evaluation_step(result_tensor, ground_truth_tensor):
    """Inserts the operations we need to evaluate the accuracy of our results.

    Args:
      result_tensor: The new final node that produces results.
      ground_truth_tensor: The node we feed ground truth data
      into.

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


def main(_):
    # Setup the directory we'll write summaries to for TensorBoard
    if tf.io.gfile.exists(FLAGS.summaries_dir):
        tf.io.gfile.rmtree(FLAGS.summaries_dir)
    tf.io.gfile.makedirs(FLAGS.summaries_dir)

    # Set up the pre-trained graph.
    maybe_download_and_extract()
    graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (
        create_inception_graph())

    # Look at the folder structure, and create lists of all the images.
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

    # See if the command-line flags mean we're applying any distortions.
    do_distort_images = should_distort_images(
        FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
        FLAGS.random_brightness)
    sess = tf.compat.v1.Session()

    if do_distort_images:
        # We will be applying distortions, so setup the operations we'll need.
        distorted_jpeg_data_tensor, distorted_image_tensor = add_input_distortions(
            FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
            FLAGS.random_brightness)
        # Pre-generate distorted bottlenecks into an in-memory pool so the
        # training loop doesn't need to re-run the full Inception network every
        # step.  augmentations_per_image controls the pool size; increase it for
        # more variety if RAM allows.
        build_distorted_bottleneck_pool(
            sess, image_lists, 'training', FLAGS.image_dir,
            distorted_jpeg_data_tensor, distorted_image_tensor,
            resized_image_tensor, bottleneck_tensor,
            augmentations_per_image=FLAGS.augmentations_per_image)
    else:
        # We'll make sure we've calculated the 'bottleneck' image summaries and
        # cached them on disk.
        cache_bottlenecks(sess, image_lists, FLAGS.image_dir, FLAGS.bottleneck_dir,
                          jpeg_data_tensor, bottleneck_tensor)

    # Add the new layer that we'll be training.
    (train_step, cross_entropy, bottleneck_input, ground_truth_input,
     final_tensor) = add_final_training_ops(len(image_lists.keys()),
                                            FLAGS.final_tensor_name,
                                            bottleneck_tensor)

    # Create the operations we need to evaluate the accuracy of our new layer.
    evaluation_step, prediction = add_evaluation_step(
        final_tensor, ground_truth_input)

    # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
    merged = tf.compat.v1.summary.merge_all()
    train_writer = tf.compat.v1.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                                   sess.graph)
    validation_writer = tf.compat.v1.summary.FileWriter(
        FLAGS.summaries_dir + '/validation')

    # Set up all our weights to their initial default values.
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    # Run the training for as many cycles as requested on the command line.
    # Start timer before training loop
    start_time = time.perf_counter()

    # Early stopping state: halt if validation accuracy hasn't improved by
    # more than early_stopping_min_delta for early_stopping_patience consecutive
    # evaluation intervals.
    best_val_accuracy = 0.0
    no_improve_count = 0
    early_stopping_patience = FLAGS.early_stopping_patience
    early_stopping_min_delta = FLAGS.early_stopping_min_delta

    for i in range(FLAGS.how_many_training_steps):
        # Get a batch of input bottleneck values, either from the pre-built pool
        # (distortion path) or from the on-disk cache (no-distortion path).
        if do_distort_images:
            train_bottlenecks, train_ground_truth = get_random_distorted_bottlenecks(
                sess, image_lists, FLAGS.train_batch_size, 'training',
                FLAGS.image_dir, distorted_jpeg_data_tensor,
                distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
        else:
            train_bottlenecks, train_ground_truth, _ = get_random_cached_bottlenecks(
                sess, image_lists, FLAGS.train_batch_size, 'training',
                FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                bottleneck_tensor)
        # Feed the bottlenecks and ground truth into the graph, and run a training
        # step. Only capture TensorBoard summaries at evaluation intervals to
        # avoid the overhead of writing summaries every step.
        is_last_step = (i + 1 == FLAGS.how_many_training_steps)
        is_eval_step = ((i % FLAGS.eval_step_interval) == 0 or is_last_step)

        if is_eval_step:
            train_summary, _ = sess.run([merged, train_step],
                                        feed_dict={bottleneck_input: train_bottlenecks,
                                                   ground_truth_input: train_ground_truth})
            train_writer.add_summary(train_summary, i)
        else:
            # Skip summary merge on non-eval steps — significantly reduces overhead.
            sess.run(train_step,
                     feed_dict={bottleneck_input: train_bottlenecks,
                                ground_truth_input: train_ground_truth})

        # Every so often, print out how well the graph is training.
        if is_eval_step:
            train_accuracy, cross_entropy_value = sess.run(
                [evaluation_step, cross_entropy],
                feed_dict={bottleneck_input: train_bottlenecks,
                           ground_truth_input: train_ground_truth})
            print('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i,
                                                            train_accuracy * 100))
            print('%s: Step %d: Cross entropy = %f' % (datetime.now(), i,
                                                       cross_entropy_value))
            validation_bottlenecks, validation_ground_truth, _ = (
                get_random_cached_bottlenecks(
                    sess, image_lists, FLAGS.validation_batch_size, 'validation',
                    FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                    bottleneck_tensor))
            # Run a validation step and capture training summaries for TensorBoard
            # with the `merged` op.
            validation_summary, validation_accuracy = sess.run(
                [merged, evaluation_step],
                feed_dict={bottleneck_input: validation_bottlenecks,
                           ground_truth_input: validation_ground_truth})
            validation_writer.add_summary(validation_summary, i)
            print('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
                  (datetime.now(), i, validation_accuracy * 100,
                   len(validation_bottlenecks)))

            # Early stopping check
            if validation_accuracy > best_val_accuracy + early_stopping_min_delta:
                best_val_accuracy = validation_accuracy
                no_improve_count = 0
            else:
                no_improve_count += 1
            if early_stopping_patience > 0 and no_improve_count >= early_stopping_patience:
                print('%s: Early stopping triggered at step %d '
                      '(no improvement for %d eval intervals). '
                      'Best validation accuracy = %.1f%%' % (
                          datetime.now(), i, no_improve_count,
                          best_val_accuracy * 100))
                break

    # Stop timer after training loop
    end_time = time.perf_counter()
    training_time = end_time - start_time
    print(f"Training Time: {training_time:.4f} seconds")

    # Save to log file
    with open("training_time_log.txt", "a") as f:
        f.write(f"{training_time:.4f}\n")

    # We've completed all our training, so run a final test evaluation on
    # some new images we haven't used before.
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

    if FLAGS.print_misclassified_test_images:
        print('=== MISCLASSIFIED TEST IMAGES ===')
        for i, test_filename in enumerate(test_filenames):
            if predictions[i] != test_ground_truth[i].argmax():
                print('%70s  %s' % (test_filename,
                                    list(image_lists.keys())[predictions[i]]))

    # Write out the trained graph and labels with the weights stored as constants.
    output_graph_def = convert_variables_to_constants(
        sess, graph.as_graph_def(), [FLAGS.final_tensor_name])
    with gfile.FastGFile(FLAGS.output_graph, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
        f.write('\n'.join(image_lists.keys()) + '\n')

# Class to calculate the F1 score on a test set after training completes, and log the results to a CSV file.
def f1_test_set_evaluation(sess, labels_list, test_dir, run_id,
                             run_number, metrics_output_dir):

    label_map = {lbl.lower().strip(): i for i, lbl in enumerate(labels_list)}
    # Gather all test samples and their true labels based on the folder structure.
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
    # Run inference on each test image and collect predictions and true labels.
    input_tensor = sess.graph.get_tensor_by_name('DecodeJpeg/contents:0') # The input tensor for raw image data
    output_tensor = sess.graph.get_tensor_by_name('final_result:0') # The output tensor for predicted probabilities

    # Loop through test samples, run inference, and collect true labels and predictions.
    y_true, y_pred = [], []
    for img_path, true_idx in samples:
        try:
            img_data = gfile.FastGFile(img_path, 'rb').read()
            predictions = sess.run(output_tensor, {input_tensor: img_data})
            y_true.append(true_idx)
            y_pred.append(int(np.argmax(predictions)))
        except Exception as e:
            print('WARNING: Could not process %s: %s' % (img_path, str(e)))

    # Calculating the F1 score, precision, and recall using the sckit-learn metrics functions. 
    f1 = round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4)
    precision = round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4)
    recall = round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4)

    print('Run %d | F1: %.4f | Precision: %.4f | Recall: %.4f' % (run_number, f1, precision, recall))
    print(classification_report(y_true, y_pred, labels=list(range(len(labels_list))), target_names=labels_list, zero_division=0))
 
    # Writes to the f1_results.csv file
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
    parser.add_argument(
        '--image_dir',
        type=str,
        default='',
        help='Path to folders of labeled images.'
    )
    parser.add_argument(
        '--output_graph',
        type=str,
        default='/tmp/output_graph.pb',
        help='Where to save the trained graph.'
    )
    parser.add_argument(
        '--output_labels',
        type=str,
        default='/tmp/output_labels.txt',
        help='Where to save the trained graph\'s labels.'
    )
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='/tmp/retrain_logs',
        help='Where to save summary logs for TensorBoard.'
    )
    parser.add_argument(
        '--how_many_training_steps',
        type=int,
        default=4000,
        help='How many training steps to run before ending.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='How large a learning rate to use when training.'
    )
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a test set.'
    )
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a validation set.'
    )
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=10,
        help='How often to evaluate the training results.'
    )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=100,
        help='How many images to train on at a time.'
    )
    parser.add_argument(
        '--test_batch_size',
        type=int,
        default=-1,
        help="""\
      How many images to test on. This test set is only used once, to evaluate
      the final accuracy of the model after training completes.
      A value of -1 causes the entire test set to be used, which leads to more
      stable results across runs.\
      """
    )
    parser.add_argument(
        '--validation_batch_size',
        type=int,
        default=100,
        help="""\
      How many images to use in an evaluation batch. This validation set is
      used much more often than the test set, and is an early indicator of how
      accurate the model is during training.
      A value of -1 causes the entire validation set to be used, which leads to
      more stable results across training iterations, but may be slower on large
      training sets.\
      """
    )
    parser.add_argument(
        '--print_misclassified_test_images',
        default=False,
        help="""\
      Whether to print out a list of all misclassified test images.\
      """,
        action='store_true'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='/tmp/imagenet',
        help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
    )
    parser.add_argument(
        '--bottleneck_dir',
        type=str,
        default='/tmp/bottleneck',
        help='Path to cache bottleneck layer values as files.'
    )
    parser.add_argument(
        '--final_tensor_name',
        type=str,
        default='final_result',
        help="""\
      The name of the output classification layer in the retrained graph.\
      """
    )
    parser.add_argument(
        '--flip_left_right',
        default=False,
        help="""\
      Whether to randomly flip half of the training images horizontally.\
      """,
        action='store_true'
    )
    parser.add_argument(
        '--random_crop',
        type=int,
        default=0,
        help="""\
      A percentage determining how much of a margin to randomly crop off the
      training images.\
      """
    )
    parser.add_argument(
        '--random_scale',
        type=int,
        default=0,
        help="""\
      A percentage determining how much to randomly scale up the size of the
      training images by.\
      """
    )
    parser.add_argument(
        '--random_brightness',
        type=int,
        default=0,
        help="""\
      A percentage determining how much to randomly multiply the training image
      input pixels up or down by.\
      """
    )
    parser.add_argument(
        '--augmentations_per_image',
        type=int,
        default=3,
        help="""\
      When distortions are enabled, how many augmented variants to pre-generate
      per training image and store in the in-memory bottleneck pool.
      Higher values give more augmentation variety at the cost of RAM and
      pool-build time. Increase if you have many GB of free RAM; decrease if
      memory is tight or if pool build time itself becomes a bottleneck.\
      """
    )
    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=10,
        help="""\
      Number of consecutive evaluation intervals with no improvement in
      validation accuracy before training is halted early.
      Set to 0 to disable early stopping entirely.\
      """
    )
    parser.add_argument(
        '--early_stopping_min_delta',
        type=float,
        default=0.005,
        help="""\
      Minimum improvement in validation accuracy (as a fraction, e.g. 0.005 = 0.5%%)
      required to reset the early-stopping patience counter.\
      """
    )
    parser.add_argument( 
        '--test_dir',
        type=str,
        default='/test_data',
        help='Directory with class-subfolder test images for F1 evaluation.'
    )
    parser.add_argument(
        '--run_id',
        type=str,
        default='baseline',
        help='Identifier for this measurement state, e.g. baseline or post_augmentation.'
    )
    parser.add_argument(
        '--eval_runs',
        type=int,
        default=5,
        help='Number of times to retrain and evaluate for F1 averaging.'
    )
    parser.add_argument(
        '--metrics_output_dir',
        type=str,
        default='../../../measurements',
        help='Directory to write F1 CSV results.'
    )
    FLAGS, unparsed = parser.parse_known_args()

    # # Reset the f1 results CSV at the start of each run
    # os.makedirs(FLAGS.metrics_output_dir, exist_ok=True)
    # csv_path = os.path.join(FLAGS.metrics_output_dir, 'f1_results.csv')
    # if os.path.isfile(csv_path):
    #     os.remove(csv_path)


#--- Reliability Score Section ---
    report = validate_directory(FLAGS.image_dir)
    print(f"Reliability Score: {report['score']:.3f} ({report['passed']}/{report['total']} valid images)")
    if report['failed']:
        print("Failed files (first 5):", report['failed'][:5])
    # Optionally, write to a file
    os.makedirs(FLAGS.metrics_output_dir, exist_ok=True)
    reliability_path = os.path.join(FLAGS.metrics_output_dir, "reliability_score.txt")
    with open(reliability_path, "w") as f:
        f.write(f"Reliability Score: {report['score']:.3f} ({report['passed']}/{report['total']} valid images)\n")
        if report['failed']:
            f.write("Failed files:\n")
            for path in report['failed']:
                f.write(path + "\n")

    # --- End Reliability Score Section ---

    # Run multiple training + evaluation cycles to get an average F1 score, since it can vary from run to run.
    all_f1, all_precision, all_recall = [], [], []

    for run_num in range(1, FLAGS.eval_runs + 1):
        print('\n=== Training + Eval Run %d/%d (run_id: %s) ===' % (
            run_num, FLAGS.eval_runs, FLAGS.run_id))
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.disable_eager_execution()
        # Call main() directly instead of via app.run() to prevent sys.exit()
        main([sys.argv[0]] + unparsed)            

 

        # Load the saved graph and evaluate F1 on the fixed test set
        eval_graph = tf.compat.v1.Graph()
        with eval_graph.as_default():
            graph_def = tf.compat.v1.GraphDef()
            with gfile.FastGFile(FLAGS.output_graph, 'rb') as f:
                graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='') 
        # Read the labels from the output_labels file to ensure correct mapping of predicted indices to class names.
        labels_list = [l.strip() for l in open(FLAGS.output_labels).readlines()]

        # Run the F1 evaluation on the test set and log the results.
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