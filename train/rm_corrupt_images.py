#!/usr/bin/python

from os import listdir, remove, path
from PIL import Image
import sys

# Check that a valid number of arguments is given.
if len(sys.argv) != 2:
    print('Invalid arguments. Usage e.g.: python3 rm_corrupt_images.py <model_name>')
    exit(1)


# The mode name.
model_name = sys.argv[1]

# The data directory paths.
all_data_dir_path = 'repo/' + model_name + '/data/all'
training_data_dir_path = 'repo/' + model_name + '/data/training'
test_data_dir_path = 'repo/' + model_name + '/data/test'

# List all directories.
img_dir_list = [all_data_dir_path, training_data_dir_path, test_data_dir_path]

# Clean the directories:
for img_dir in img_dir_list:

    # Check if directory exists.
    if path.exists(img_dir):

        # Go through all image files to split them as either Training or Validation data.
        for label_dir_name in listdir(img_dir):

            label_dir_path =  img_dir + '/' + label_dir_name

            print('---------------------------------------------')
            print('Scanning ' + label_dir_path)

            total_image_counter = 0
            invalid_image_counter = 0

            # Go through the image files.
            for f in listdir(label_dir_path):

                # Increment image counter.
                total_image_counter = total_image_counter + 1

                # Delete file if it can't be opened.
                img_file_path = label_dir_path + '/' + f

                try:
                    im = Image.open(img_file_path)
                    
                except:
                    # Count invalid image.
                    invalid_image_counter = invalid_image_counter + 1
                    
                    # Verbosity on which image will be removed.
                    print('Bad image will be removed: ' + f)
                    
                    # Delete the images.
                    remove(img_file_path)
                

            print(str(invalid_image_counter) + '/' + str(total_image_counter) + ' corrupt images were removed.\n')