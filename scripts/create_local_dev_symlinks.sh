#!/bin/bash

# This script creates the TensorFlow Lite shared object symlinks
# necessary to run this app in local development environment.
# 
# Run this bash script from the project's home directory: 
# ./scripts/create_local_dev_symlinks.sh 

# Architecture used to build TensorFlow Lite shared objects for the
# local development environment. We want to replace "armhf" in the 
# symbolic link commands taken from postinst script to this arch value.
arch="k8"

# Get the project directory's path.
pwd=$(pwd)

# Read the postinst ipk post installation bash script line by line.
while IFS= read -r line
do
    # For each symbolic link creation command line in postinst...
    if  [[ $line == $'ln -s '* ]] ;
    then
        # Replace arch in the file path folder names from armhf to $arch.
        line=${line//armhf/"$arch"}

        # Replace the path to that of the development environment in this
        # local dev environment.
        line=${line//" /"/" $pwd/"}

        # Execute the symbolic link creation command for this local dev environment.
        eval "$line"
    fi
done < "sepp_package/CONTROL/postinst"
