#!/usr/bin/python

import sys
import os
import shutil

# Check that a valid number of arguments is given.
if len(sys.argv) != 2:
    print('Invalid arguments. Usage e.g.: python3 batch_label_images.py bad_orientation')

# The name of the model that is going to be trained.
model_name = sys.argv[1]

# Directory and file paths.
DIR_CLASSIFICATION = 'repo/' + model_name + '/data/classification'
DIR_CLASSIFIED_IMG = DIR_CLASSIFICATION + '/classified'
DIR_UNCLASSIFIED_IMG = DIR_CLASSIFICATION + '/unclassified'
CSV_FILENAME = DIR_CLASSIFICATION + '/confidences.csv'

# Clear prediction logs from previous validation.
if os.path.exists(CSV_FILENAME):
    os.remove(CSV_FILENAME) 

# The path of the validation data directory.
validation_data_dir_path = 'repo/' + model_name + '/data/validation'

# Go through all image files to split them as either Training or Validation data.
for label_dir_name in os.listdir(validation_data_dir_path):
    print("Validating against '" + label_dir_name + "' images...")

    # The label directory path.
    label_dir_path = validation_data_dir_path + '/' + label_dir_name

    # Delete labels directory if already exists from previous validation.
    shutil.rmtree(DIR_CLASSIFIED_IMG + '/' + label_dir_name, ignore_errors=True)
    shutil.rmtree(DIR_UNCLASSIFIED_IMG + '/' + label_dir_name, ignore_errors=True) 

    # Go through each Validation image in the current label directory and predict its label.
    for image_file in os.listdir(label_dir_path):
        # The image file path.
        image_path = label_dir_path + '/' + image_file

        # Build the python command to executing the label_image.py prediction/inference script that feeds the image into the trained mode.
        # Default values are used for parameters that are not set, see label_image.py for what those parameters and default values are.
        cmd_label_image = 'python3 label_image.py -n {0} -i {1} -m repo/{0}/tflite_model.tflite -l repo/{0}/labels.txt -e {2}'\
            .format(model_name, image_path, label_dir_name)

        # Predict!
        os.system(cmd_label_image)