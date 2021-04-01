#!/bin/bash

# Check that expected number of parameters is given.
if [ "$#" -ne 2 ]; then
    echo "Invalid number of parameters. Usage e.g.: ./train_model.sh <model_name> <epoch_num>"
    exit 1
fi

# Delete summary files created from previous training session.
if [ -d "repo/$1/summaries" ] 
then
    rm -rf repo/$1/summaries
fi

# Train a model.
make_image_classifier \
  --image_dir repo/$1/data/training \
  --tfhub_module https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4 \
  --saved_model_dir repo/$1 \
  --labels_output_file repo/$1/labels.txt \
  --tflite_output_file repo/$1/tflite_model.tflite \
  --train_epochs $2 \
  --summaries_dir repo/$1/summaries