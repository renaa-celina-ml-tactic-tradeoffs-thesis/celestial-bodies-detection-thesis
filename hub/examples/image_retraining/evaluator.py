# evaluator.py
# ML Lifecycle Stage: Evaluation
# Responsible for post-training evaluation: computing F1, precision, and
# recall on the held-out test set, logging results to CSV, and computing
# the dataset reliability score.
# No training logic, no model building, no export logic lives here.

import os
import csv
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             classification_report)


def f1_test_set_evaluation(sess, labels_list, test_dir, run_id,
                            run_number, metrics_output_dir):
    """Computes F1, precision, and recall on the fixed test set.

    Runs inference on every image in test_dir using the loaded graph,
    then computes weighted classification metrics and logs to CSV.

    Args:
      sess: Active TensorFlow Session with the trained graph loaded.
      labels_list: List of class label strings in index order.
      test_dir: Path to directory with class-named subfolders of test images.
      run_id: String identifier for this measurement state.
      run_number: Integer run index for logging.
      metrics_output_dir: Directory path to write the CSV output.

    Returns:
      Tuple of (f1, precision, recall) as floats.
    """
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


def validate_directory(image_dir, allowedexts=('.jpg', '.jpeg', '.JPG', '.JPEG'),
                       verbose=False):
    """Checks all images in a directory tree for validity.

    Validates extension, non-zero file size, and JPEG decodability.

    Args:
      image_dir: Root directory to walk.
      allowedexts: Tuple of accepted file extensions.
      verbose: If True, prints each failing file path.

    Returns:
      Dict with keys: total, passed, failed (list), score (float 0–1).
    """
    total = 0
    passed = 0
    failed = []

    for root, _, files in os.walk(image_dir):
        for fname in files:
            if not fname.endswith(allowedexts):
                continue
            fpath = os.path.join(root, fname)
            total += 1
            if os.path.getsize(fpath) == 0:
                failed.append(fpath)
                if verbose:
                    print(f"Zero size: {fpath}")
                continue
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
