# exporter.py
# ML Lifecycle Stage: Deployment / Export
# Responsible for downloading the pre-trained Inception v3 model if needed,
# loading the frozen graph, and exporting the retrained graph as a frozen .pb
# file ready for inference deployment.
# No training logic, no evaluation logic, no data loading lives here.

import os
import sys
import tarfile

import tensorflow as tf
from six.moves import urllib
from tensorflow.compat.v1.graph_util import convert_variables_to_constants
from tensorflow.python.platform import gfile

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'


def maybe_download_and_extract(model_dir):
    """Downloads and extracts the Inception v3 model tar file if not present.

    Args:
      model_dir: Directory path to store the downloaded model.
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(model_dir, filename)

    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    tarfile.open(filepath, 'r:gz').extractall(model_dir)


def create_inception_graph(model_dir):
    """Creates a graph from the saved Inception v3 GraphDef file.

    Args:
      model_dir: Directory containing classify_image_graph_def.pb.

    Returns:
      Tuple of (graph, bottleneck_tensor, jpeg_data_tensor,
                resized_input_tensor).
    """
    with tf.compat.v1.Session() as sess:
        model_filename = os.path.join(model_dir, 'classify_image_graph_def.pb')
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
                tf.import_graph_def(graph_def, name='', return_elements=[
                    BOTTLENECK_TENSOR_NAME,
                    JPEG_DATA_TENSOR_NAME,
                    RESIZED_INPUT_TENSOR_NAME]))
    return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor


def export_model(sess, graph, final_tensor_name, output_graph_path,
                 output_labels_path, image_lists):
    """Exports the retrained graph as a frozen .pb file with label list.

    Converts all variables to constants and writes the frozen graph to disk,
    alongside a plain-text label file.

    Args:
      sess: Active TensorFlow Session containing the trained weights.
      graph: The TensorFlow graph object.
      final_tensor_name: Name of the output classification tensor.
      output_graph_path: File path to write the frozen .pb graph.
      output_labels_path: File path to write the labels text file.
      image_lists: Dictionary of image lists (used to extract label names).
    """
    output_graph_def = convert_variables_to_constants(
        sess, graph.as_graph_def(), [final_tensor_name])
    with gfile.FastGFile(output_graph_path, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    with gfile.FastGFile(output_labels_path, 'w') as f:
        f.write('\n'.join(image_lists.keys()) + '\n')
    print(f"Model exported to {output_graph_path}")
    print(f"Labels exported to {output_labels_path}")
    