#!/bin/sh

# Check if the app is running.
result=`ps aux | grep -i "smartcam.py" | grep -v "grep" | wc -l`

if [ $result -ge 1 ]
    then
        touch .stop
fi

# exit
exit 0