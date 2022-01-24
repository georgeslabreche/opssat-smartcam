#!/bin/sh

# Check if the app is already running.
result=`ps aux | grep -i "smartcam.py" | grep -v "grep" | wc -l`
if [ $result -ge 1 ]
    then
        # Exit if the app is already running.
        exit 0
    else
        # Run the app if it's not already running.
        /home/exp1000/smartcam.py
        exit 0
fi