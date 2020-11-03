#!/bin/sh

# Check if the app is already running.
result=`ps aux | grep -i "acquire_and_label_images.py" | grep -v "grep" | wc -l`
if [ $result -ge 1 ]
    then
        # Exit if the app is already running.
        exit 0
    else
        # Run the app if it's not already running.
        /home/exp1000/acquire_and_label_images.py
        exit 0
fi