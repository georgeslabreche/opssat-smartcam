#!/usr/bin/env bash

# The project directory path.
# Remove the scripts folder in case this bash script is being executed from the scripts folder
# instead of from the project root folder.
project_dir=$(pwd)
project_dir=${project_dir/scripts/""}

# Deploy the ipk file to the SEPP on the EM.
scp -P2223 *.ipk  root@localhost:/home/root/georges/apps