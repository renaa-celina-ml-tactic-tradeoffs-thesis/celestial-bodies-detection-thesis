# data_loader.py
# ML Lifecycle Stage: Data Ingestion

import os
import re
import hashlib

from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


def _assign_split(file_name, testing_percentage, validation_percentage):
    """Determines which split a file belongs to using its filename hash.
    
    Uses a stable SHA-1 hash so the same file always lands in the same
    split, even if the dataset grows later.

    Args:
      file_name: Full file path string.
      testing_percentage: Integer percentage reserved for testing.
      validation_percentage: Integer percentage reserved for validation.

    Returns:
      String: 'training', 'testing', or 'validation'.
    """
    hash_name = re.sub(r'_nohash_.*$', '', file_name)
    hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                        (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                       (100.0 / MAX_NUM_IMAGES_PER_CLASS))

    if percentage_hash < validation_percentage:
        return 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        return 'testing'
    else:
        return 'training'


def _collect_files_for_class(image_dir, dir_name):
    """Collects all JPEG file paths for a single class subfolder.

    Args:
      image_dir: Root image directory string.
      dir_name: Name of the class subfolder.

    Returns:
      List of file path strings, or empty list if none found.
    """
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    file_list = []
    for extension in extensions:
        file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
        file_list.extend(gfile.Glob(file_glob))

    if not file_list:
        print('No files found')
    elif len(file_list) < 20:
        print('WARNING: Folder has less than 20 images, which may cause issues.')
    elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
        print('WARNING: Folder {} has more than {} images. Some images will '
              'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))

    return file_list


def create_image_lists(image_dir, testing_percentage, validation_percentage):
    """Builds a list of training images from the file system.

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

        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue

        print("Looking for images in '" + dir_name + "'")
        file_list = _collect_files_for_class(image_dir, dir_name)
        if not file_list:
            continue

        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        splits = {'training': [], 'testing': [], 'validation': []}

        for file_name in file_list:
            base_name = os.path.basename(file_name)
            split = _assign_split(file_name, testing_percentage, validation_percentage)
            splits[split].append(base_name)

        result[label_name] = {
            'dir': dir_name,
            'training': splits['training'],
            'testing': splits['testing'],
            'validation': splits['validation'],
        }

    return result


def get_image_path(image_lists, label_name, index, image_dir, category):
    """Returns a path to an image for a label at the given index."""
    import tensorflow as tf
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
    return get_image_path(
        image_lists, label_name, index, bottleneck_dir, category) + '.txt'
