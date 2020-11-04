#!/bin/sh

# Check if the app is already running.
result=`ps aux | grep -i "acquire_and_label_images.py" | grep -v "grep" | wc -l`
if [ $result -ge 1 ]
    then
        # TODO: Kill the app process if it is running.
        exit 0
    else
        # Exit if the app is not running.
        exit 0
fi