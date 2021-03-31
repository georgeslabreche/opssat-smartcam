
#!/bin/bash

# Check that expected number of parameters is given.
if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters. Usage e.g.: ./create_dirs.sh <model_name>"
    exit 1
fi

# Create directories.
echo "Creating directories to train and test the '$1' model..."
mkdir -p repo/$1
mkdir -p repo/$1/summaries
mkdir -p repo/$1/data/all
mkdir -p repo/$1/data/training
mkdir -p repo/$1/data/test
mkdir -p repo/$1/data/classification
mkdir -p repo/$1/data/classification/classified
mkdir -p repo/$1/data/classification/unclassified

echo "Done."
echo "Next step: move your training and test images into the training and test directories."
echo "Read the README for further instructions."

