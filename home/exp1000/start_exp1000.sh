#!/bin/sh

# FIXME: Sould it be !/bin/sh for busybox?

# Check if the app is already running.
result=`ps aux | grep -i "label_images.py" | grep -v "grep" | wc -l`
if [ $result -ge 1 ]
    then
        # Exit if the app is already running.
        exit 0
    else
        # Run the app if it's not already running.
        #/home/exp1000/label_images.py
        /home/georges/apps/SmartCamLuvsU/home/exp1000/label_images.py
        exit 0
fi