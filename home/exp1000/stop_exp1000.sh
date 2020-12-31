#!/bin/sh

# Check if the app is already running.
result=`ps aux | grep -i "acquire_and_label_images.py" | grep -v "grep" | wc -l`
if [ $result -ge 1 ]
    then
        # Kill the app process if it is running.
        kill $(ps aux | grep -i "acquire_and_label_images.py" | grep -v "grep" | awk '{ print $1 }')

        # Kill the image classification program in case it was triggered before killing the app
        kill $(ps aux | grep -i "image_classifier" | grep -v "grep" | head -n1 | awk '{ print $1 }')
        kill $(ps aux | grep -i "image_classifier" | grep -v "grep" | head -n2 | tail -1 | awk '{ print $1 }')

        # Delete temporary files if they exist.
        rm -f *.ims_rgb
        rm -f *.png
        rm -f *.jpeg
        rm -f *.tar
        rm -f *.tar.gz

        # Exit.
        exit 0
        
    else
        # Just exit if the app is not running.
        exit 0
fi