#!/usr/bin/python

import sys
import os
from os import listdir
import shutil
import math

# Check that a valid number of arguments is given.
if len(sys.argv) != 3:
    print('Invalid arguments. Usage e.g.: python3 split_data.py <model_name> 25')

# The name of the model that is going to be trained.
model_name = sys.argv[1]

# The percentage of data to use as validation data.
split_percent = int(sys.argv[2])

# The split modulo to determine the splitting.
split_modulo = math.ceil(100 / split_percent)

# The data directory paths.
all_data_dir_path = 'repo/' + model_name + '/data/all'
training_data_dir_path = 'repo/' + model_name + '/data/training'
validation_data_dir_path = 'repo/' + model_name + '/data/validation'

# Use this counter to track number of images files.
img_counter = 0

# Go through all image files to split them as either Training or Validation data.
for label_dir_name in listdir(all_data_dir_path):
    print("Splitting '" + label_dir_name + "' images into Training or Validation datasets...")

    # The label directory path.
    label_dir_path = all_data_dir_path + "/" + label_dir_name

    # Create the label folder in the Training directory, if it doesn't exist already.
    if not os.path.exists(training_data_dir_path + '/' + label_dir_name):
        os.makedirs(training_data_dir_path + '/' + label_dir_name)

    # Create the label folder in the Validation directory, if it doesn't exist already.
    if not os.path.exists(validation_data_dir_path + '/' + label_dir_name):
        os.makedirs(validation_data_dir_path + '/' + label_dir_name)

    # Go through each image in the current label directory and copy it to either the Training or Validation directory.
    for image_file in listdir(label_dir_path):

        # Use this counter to track the split.
        img_counter = img_counter + 1
        
        # Use module to split files between Training and Validation.
        if img_counter % split_modulo == 0:
            # This image is copied to the Validation directory.
            shutil.copyfile(label_dir_path + '/' + image_file, validation_data_dir_path + '/' + label_dir_name + '/' + image_file)
        else:
            # This image is copied to the Training directory.
            shutil.copyfile(label_dir_path + '/' + image_file, training_data_dir_path + '/' + label_dir_name + '/' + image_file)

# Done.
print('Done')