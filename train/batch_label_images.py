#!/usr/bin/python

import sys
import os
import shutil
import progressbar

# Check that a valid number of arguments is given.
#   argv[1] --> model name.
#   argv[2] --> labels to skip. 
if len(sys.argv) not in [2, 3]:
    print('Invalid arguments. Usage e.g.:')
    print('  python3 batch_label_images.py my_model_name')
    print('  python3 batch_label_images.py my_model_name skip_label_1,skip_label_2')

# The name of the model that is going to be trained.
model_name = sys.argv[1]

# The model file.
model_file = 'repo/{0}/tflite_model.tflite'.format(model_name)

# The label file.
label_file = 'repo/{0}/labels.txt'.format(model_name)

# Check if model file exists.
if not os.path.exists(model_file):
    print("The model file doesn't exits: " + model_file)
    exit(1)

# Check if label file exists.
if not os.path.exists(label_file):
    print("The label file doesn't exits: " + label_file)
    exit(1)

# Skip testing for these labels.
labels_skip_list = sys.argv[2].split(",") if len(sys.argv) == 3 else []

# Directory and file paths.
DIR_CLASSIFICATION = 'repo/' + model_name + '/data/classification'
DIR_CLASSIFIED_IMG = DIR_CLASSIFICATION + '/classified'
DIR_UNCLASSIFIED_IMG = DIR_CLASSIFICATION + '/unclassified'
CSV_FILENAME = DIR_CLASSIFICATION + '/confidences.csv'

# Clear prediction logs from previous test.
if os.path.exists(CSV_FILENAME):
    os.remove(CSV_FILENAME) 

# The path of the Test data directory.
test_data_dir_path = 'repo/' + model_name + '/data/test'

# Go through all image files to split them as either Training or Test data.
for label_dir_name in os.listdir(test_data_dir_path):

    if label_dir_name in labels_skip_list:
        print("Skip testing against '" + label_dir_name + "' images.")

    else:
        print("Testing model with '" + label_dir_name + "' images...") 

        # The label directory path.
        label_dir_path = test_data_dir_path + '/' + label_dir_name

        # Delete classification directories if already exists from previous test.
        shutil.rmtree(DIR_CLASSIFIED_IMG + '/' + label_dir_name, ignore_errors=True)
        shutil.rmtree(DIR_UNCLASSIFIED_IMG + '/' + label_dir_name, ignore_errors=True) 

        # Go through each Test image in the current label directory and predict its label.
        for image_file in progressbar.progressbar(os.listdir(label_dir_path), redirect_stdout=True):
            # The image file path.
            image_path = label_dir_path + '/' + image_file

            # Build the python command to executing the label_image.py prediction/inference script that feeds the image into the trained mode.
            # Default values are used for parameters that are not set, see label_image.py for what those parameters and default values are.
            cmd_label_image = 'python3 label_image.py -n {0} -i {1} -m {2} -l {3} -e {4} > /dev/null 2>&1'\
                .format(model_name, image_path, model_file, label_file, label_dir_name)

            # Predict!
            os.system(cmd_label_image)