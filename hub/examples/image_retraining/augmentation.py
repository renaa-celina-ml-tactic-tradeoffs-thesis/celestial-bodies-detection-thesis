# augmentation.py
# ML Lifecycle Stage: Data Preprocessing
# Responsible for building the TensorFlow distortion graph that applies
# random augmentations to training images (flipping, cropping, scaling,
# brightness). Also handles distorted bottleneck retrieval during training.
# No caching logic, no evaluation logic, no export logic lives here.

import random

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile

from data_loader import get_image_path, MAX_NUM_IMAGES_PER_CLASS
from bottleneck_cache import run_bottleneck_on_image

MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3


def should_distort_images(flip_left_right, random_crop, random_scale, random_brightness):
    """Returns True if any distortion flags are enabled.

    Args:
      flip_left_right: Boolean whether to randomly mirror images horizontally.
      random_crop: Integer percentage margin for random cropping.
      random_scale: Integer percentage of how much to vary scale by.
      random_brightness: Integer range to randomly multiply pixel values by.

    Returns:
      Boolean value indicating whether any distortions should be applied.
    """
    return (flip_left_right or (random_crop != 0) or
            (random_scale != 0) or (random_brightness != 0))


def add_input_distortions(flip_left_right, random_crop, random_scale, random_brightness):
    """Creates the TensorFlow operations to apply the specified distortions.

    Args:
      flip_left_right: Boolean whether to randomly mirror images horizontally.
      random_crop: Integer percentage setting the total margin around the crop box.
      random_scale: Integer percentage of how much to vary the scale by.
      random_brightness: Integer range to randomly multiply pixel values by.

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
    resize_scale_value = tf.random.uniform(tensor_shape.scalar(),
                                           minval=1.0, maxval=resize_scale)
    scale_value = tf.multiply(margin_scale_value, resize_scale_value)
    precrop_width = tf.multiply(scale_value, MODEL_INPUT_WIDTH)
    precrop_height = tf.multiply(scale_value, MODEL_INPUT_HEIGHT)
    precrop_shape = tf.stack([precrop_height, precrop_width])
    precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
    precropped_image = tf.image.resize(decoded_image_4d, precrop_shape_as_int,
                                       method=tf.image.ResizeMethod.BILINEAR)
    precropped_image_3d = tf.squeeze(precropped_image, axis=[0])
    cropped_image = tf.image.random_crop(
        precropped_image_3d,
        [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, MODEL_INPUT_DEPTH])

    if flip_left_right:
        flipped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        flipped_image = cropped_image

    brightness_min = 1.0 - (random_brightness / 100.0)
    brightness_max = 1.0 + (random_brightness / 100.0)
    brightness_value = tf.random.uniform(tensor_shape.scalar(),
                                         minval=brightness_min,
                                         maxval=brightness_max)
    brightened_image = tf.multiply(flipped_image, brightness_value)
    distort_result = tf.expand_dims(brightened_image, 0, name='DistortResult')
    return jpeg_data, distort_result


def get_random_distorted_bottlenecks(sess, image_lists, how_many, category,
                                     image_dir, input_jpeg_tensor, distorted_image,
                                     resized_input_tensor, bottleneck_tensor):
    """Retrieves bottleneck values for training images after applying distortions.

    Used when distortions are enabled and cached bottlenecks cannot be reused.

    Args:
      sess: Current TensorFlow Session.
      image_lists: Dictionary of training images for each label.
      how_many: Integer number of bottleneck values to return.
      category: Name string of which set of images to fetch.
      image_dir: Root folder string of the subfolders containing the images.
      input_jpeg_tensor: The input layer we feed image data to.
      distorted_image: The output node of the distortion graph.
      resized_input_tensor: The input node of the recognition graph.
      bottleneck_tensor: The bottleneck output layer of the CNN graph.

    Returns:
      List of bottleneck arrays and their corresponding ground truths.
    """
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []

    for unused_i in range(how_many):
        label_index = random.randrange(class_count)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
        image_path = get_image_path(image_lists, label_name, image_index,
                                    image_dir, category)
        if not gfile.Exists(image_path):
            tf.compat.v1.logging.fatal('File does not exist %s', image_path)
        jpeg_data = gfile.FastGFile(image_path, 'rb').read()
        distorted_image_data = sess.run(distorted_image, {input_jpeg_tensor: jpeg_data})
        bottleneck = run_bottleneck_on_image(sess, distorted_image_data,
                                             resized_input_tensor, bottleneck_tensor)
        ground_truth = np.zeros(class_count, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)

    return bottlenecks, ground_truths