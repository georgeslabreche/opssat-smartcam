# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

# Taken from: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
# Modified for the OPS-SAT SmartCam app.

"""label_image for tflite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
from PIL import Image
import tensorflow as tf # TF2

import shutil
import ntpath
from pathlib import Path

import csv

# The label columns will be dynamically added later on based on the training data classifications.
CSV_COLUMNS = ['filename', 'time_ms', 'predicted_label', 'expected_label']

def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-t',
      '--threshold',
      default=0, type=float,
      help='confidence threshold for a valid prediction')
  parser.add_argument(
      '-n',
      '--model_name',
      help='nam of model')
  parser.add_argument(
      '-i',
      '--image',
      help='image to be classified')
  parser.add_argument(
      '-m',
      '--model_file',
      help='.tflite model to be executed')
  parser.add_argument(
      '-l',
      '--label_file',
      help='name of file containing labels')
  parser.add_argument(
      '--input_mean',
      default=0, type=float,
      help='input_mean')
  parser.add_argument(
      '--input_std',
      default=255, type=float,
      help='input standard deviation')
  parser.add_argument(
      '-e',
      '--expected_label',
      help='expected label')
  parser.add_argument(
      '--num_threads', default=None, type=int, help='number of threads')
  args = parser.parse_args()

  # Path of classified folder where to copy image file based on label prediction confidence level.
  DIR_CLASSIFIED_IMG = 'repo/' + args.model_name + '/data/classification/classified'

  # Path of unclassified folder where to copy image file if labeling confidence level did not pass threshold.
  DIR_UNCLASSIFIED_IMG = 'repo/' + args.model_name + '/data/classification/unclassified'

  # Path of CSV files to log prediction confidence values.
  CSV_FILENAME = 'repo/' + args.model_name + '/data/classification/confidences.csv'
  
  # Prepare CSV row for file processing report.
  csv_row = {}
  csv_row['filename'] = ntpath.basename(args.image)

  interpreter = tf.lite.Interpreter(
      model_path=args.model_file, num_threads=args.num_threads)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # check the type of the input tensor
  floating_model = input_details[0]['dtype'] == np.float32

  # NxHxWxC, H:1, W:2
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]
  img = Image.open(args.image).resize((width, height))

  # add N dim
  input_data = np.expand_dims(img, axis=0)

  if floating_model:
    input_data = (np.float32(input_data) - args.input_mean) / args.input_std

  interpreter.set_tensor(input_details[0]['index'], input_data)

  start_time = time.time()
  interpreter.invoke()
  stop_time = time.time()

  output_data = interpreter.get_tensor(output_details[0]['index'])
  results = np.squeeze(output_data)

  top_k = results.argsort()[-5:][::-1]
  labels = load_labels(args.label_file)
  
  # Some pre-processing for outputs.
  for lbl in labels:
    # Make sure the label directories exist.
    Path(DIR_CLASSIFIED_IMG + '/' + lbl).mkdir(parents=True, exist_ok=True)
    
    # Include label as a column in the confidence report CSV file.
    if lbl not in CSV_COLUMNS:
      CSV_COLUMNS.append(lbl)
        
  
  # A flag to mark if an image has been classified or not.
  classified = False

  csv_row['expected_label'] = args.expected_label
  
  # Loop through the labels and classify images based on confidence threshold.
  for i in top_k:

    # Determine result.
    result = 0
    
    if floating_model:
      result = float(results[i])
      
    else:
      result = float(results[i] / 255.0)
      
    # Print result.  
    print('{:08.6f}: {}'.format(result, labels[i]))
    
    # Write confidence result in CSV report.
    csv_row[labels[i]] = result
    
    # Check if result is of high enough confidence to classify the image.
    if result >= args.threshold:
    
        # Mark image as classified.
        classified = True

        # Copy image to classified directory (if above a confidence threshold).
        dst_filename = DIR_CLASSIFIED_IMG + '/' + labels[i] + '/' + ntpath.basename(args.image)
        shutil.copyfile(args.image, dst_filename)

        # In CSV report indicate which classification was applied.
        csv_row['predicted_label'] = labels[i]
   
  # If image was not classified in either of the labels then copy it into an unclassified directory.
  if not classified:
    dst_filename = DIR_UNCLASSIFIED_IMG + '/' + ntpath.basename(args.image)
    shutil.copyfile(args.image, dst_filename)
    
    # Indicate this case in the CSV report.
    csv_row['predicted_label'] = 'n/a'

  # Calculate processing time.
  processing_time = (stop_time - start_time) * 1000
  
  # Print processing time.
  print('time: {:.3f}ms'.format(processing_time))
  
  # Include processing time in CSV report.
  csv_row['time_ms'] = processing_time
  
  # If CSV file does not exist, write it and include the header columns.
  existing_csv_file = Path(CSV_FILENAME)
  if not existing_csv_file.is_file():
      with open(CSV_FILENAME, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
  
  # Write classification confidence rows in CSV file.
  with open(CSV_FILENAME, 'a+') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    writer.writerow(csv_row)