#!/bin/sh

# Check if the app is running.
result=`ps aux | grep -i "acquire_and_label_images.py" | grep -v "grep" | wc -l`

if [ $result -ge 1 ]
    then
        touch .stop
fi

# exit
exit 0