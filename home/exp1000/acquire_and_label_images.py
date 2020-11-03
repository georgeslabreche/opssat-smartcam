#!/usr/bin/python3

import platform
import os
import subprocess
import glob
import configparser
import logging
import time
import datetime
import json
import operator

__author__ = 'Georges Labreche, Georges.Labreche@esa.int'

# The experiment id number.
exp_id = 1000

# Select production or development environment dev path based on platform name.
base_path = '/home/root/georges/apps/SmartCamLuvsU/home/exp' + str(exp_id) if platform.node() == 'sepp' else '/home/georges/dev/SmartCamLuvsU/home/exp' + str(exp_id) 

# The experiment's config file path.
config_file = base_path + '/config.sepp.ini' if platform.node() == 'sepp' else 'config.dev.ini'

# The experiment's log folder path.
log_path = base_path + '/logs'

# The experiment's toGround folder path.
toGround_path = base_path + '/toGround'

# The filestore's toGround folder path.
filstore_toGroud = '/home/root/esoc-apps/fms/filestore/toGround'

# Image classifier program file path.
program_path = base_path + '/bin/tensorflow/lite/c/image_classifier'

def remove_images(logger):
    """Delete image files created on the project's root directory while processing the acquired image.
    These files could exist due to an unhandled error during a previous run so as a precaution we also run this function prior to image acquisition.
    """

    # Count the number of files deleted.
    delete_count = 0

    # Loop through all the file types to delete.
    for ext in ['ims_rgb', 'png', 'jpeg']:

        # Get a list of all the file paths that ends with .txt from in specified directory
        img_files = glob.glob(base_path + "/*." + ext)

        for f in img_files:
            try:
                os.remove(f)
                delete_count = delete_count + 1
                logger.info("Removed image file: " + f)

            except:
                logger.error("Error removing image file: " + f)
                return -1

    return delete_count


def main(startTime, logger, logfile):
    """Run the experiment."""

    # Init the config parser, read the config file.
    config = configparser.ConfigParser()
    config.read(config_file)

    # Determine if this is a dry run or not
    dry_run = config.getboolean('conf', 'dry_run')

    # Fetch model config values.
    tflite_model = config.get('model', 'tflite_model')
    file_labels = config.get('model', 'labels')
    labels_keep = json.loads(config.get('model', 'labels_keep'))
    input_height = config.get('model', 'input_height')
    input_width = config.get('model', 'input_width')
    input_mean = config.get('model', 'input_mean')
    input_std = config.get('model', 'input_std')
    confidence_threshold = config.get('model', 'confidence_threshold')

    # Fetch image acquisition parameters.
    gen_interval = config.getint('gen', 'gen_interval')
    
    gen_number = config.getint('gen', 'gen_number')
    if gen_number <= 0:
        gen_number = 1

    gen_exposure = config.getint('gen', 'gen_exposure')
    if gen_number <= 1:
        gen_exposure = 2

    gen_gains = json.loads(config.get('gen', 'gen_gains'))
    if gen_gains[0] >= 255:
        gen_gains[0] = 255

    if gen_gains[1] >= 255:
        gen_gains[1] = 255

    if gen_gains[2] >= 255:
        gen_gains[2] = 255

    # Fetch image file retention parameters.
    raw_keep = config.getboolean('img', 'raw_keep')
    png_keep = config.getboolean('img', 'png_keep')
    log_keep = config.getboolean('img', 'log_keep')

    # Fetch jpeg thumbnail image processing parameters.
    jpeg_scaling = config.getfloat('jpeg', 'jpeg_scaling')
    if jpeg_scaling >= 1.0 or jpeg_scaling <= 0:
        jpeg_scaling = 0.5  

    jpeg_quality  = config.getint('jpeg', 'jpeg_quality')
    if jpeg_quality >= 100 or jpeg_quality <= 0:
        jpeg_quality = 90

    jpeg_processing = config.get('jpeg', 'jpeg_processing')
    if jpeg_processing != 'pnmnorm' and jpeg_processing != 'pnmhisteq':
        jpeg_processing = 'none'

    # Flag and counter to keep track
    done = False
    counter = 0

    while not done:
        try:

            skip = False

            # Cleanup any files that may have been left over from a previous run that may have terminated ungracefully.
            if remove_images(logger) < 0:
                skip = True

            # If experiment's root directory is clean, i.e. no images left over from a previous image acquisition, then acquire a new image.
            if not skip:
            
                # Build the image acquisition execution command string.
                cmd_image_acquisition = 'ims100_testapp -R {R} -G {G} -B {B} -c /dev/ttyACM0 -m /dev/sda -v 0 -n 1 -p -e {E} >> {L} 2>&1'.format(\
                    R=gen_gains[0],\
                    G=gen_gains[1],\
                    B=gen_gains[2],\
                    E=gen_exposure,\
                    L=logfile)
                
                # Log the command that will be executed.
                logger.info("Running command to acquire an image: {C}".format(C=cmd_image_acquisition))

                # Run the image acquisition command. 
                os.system(cmd_image_acquisition) if not dry_run else print(cmd_image_acquisition)

                # Check that png file exists...
                png_files = glob.glob(base_path + "/*.png")
                
                # If the png file doesn't exist then skip this iteration.
                if len(png_files) != 1:
                    logger.error("Failed to acquire an image from the camera.")
                    skip = True
                else:
                    file_png = png_files[0]
                    logger.info("Acquired image: " + file_png)
            
            # If we have successfully acquired a png file then proceed with creating the jpeg thumbnail.
            if not skip:

                file_thumbnail = file_png.replace(".png", "_thumbnail.jpeg")

                # Build the thumbnail creation command string.
                if jpeg_processing != 'none':
                    cmd_create_thumbnail = 'pngtopam {F} | pamscale {S} | {P} | pnmtojpeg -quality {Q} > {O}'.format(\
                        S=jpeg_scaling,\
                        Q=jpeg_quality,\
                        P=jpeg_processing,\
                        F=file_png,\
                        O=file_thumbnail)

                else:
                    cmd_create_thumbnail = 'pngtopam {F} | pamscale {S} | pnmtojpeg -quality {Q} > {O}'.format(\
                        S=jpeg_scaling,\
                        Q=jpeg_quality,\
                        F=file_png,\
                        O=file_thumbnail)

                # Log the command that will be executed.
                logger.info("Running command to create thumbnail: {C}".format(C=cmd_create_thumbnail))
            
                # Run the thumbnail creation command.
                os.system(cmd_create_thumbnail) if not dry_run else print(cmd_create_thumbnail)

                # Check that thumbnail exists.
                if not os.path.isfile(file_thumbnail):
                    logger.error("Failed to generate a thumbnail.")
                    skip = True

            # If we have successfully created a jpeg thumbnail file then create the jpeg input file for the image classification program.
            if not skip:

                # File name of the image file that will be used as the input image to feed the image classification model.
                file_image_input = file_thumbnail.replace("_thumbnail.jpeg", "_input.jpeg")

                # Build the command string to create the image input for the image classification program.
                cmd_create_input_image = 'pamscale -xsize {X} -ysize {Y} > {O}'.format(\
                        X=input_width,\
                        Y=input_height,\
                        O=file_image_input)

                # Log the command that will be executed.
                logger.info("Running command to create input: {C}".format(C=cmd_create_input_image))

                # Run the command to create the image input file for the image classification program.
                os.system(cmd_create_input_image) if not dry_run else print(cmd_create_input_image)

                # Check that the image input exists.
                if not os.path.isfile(file_image_input):
                    logger.error(\
                        "Failed to generate {X}x{Y} image input for the image classification model.".format(\
                            X=input_width,
                            Y=input_height))

                    skip = True

            # If we have successfully create the input images then feed it into the iamge classification program.
            if not skip:

                # Build the image labelling command.
                cmd_label_image = '{P} {I} {M} {L} {height} {width} {mean} {std}'.format(\
                    P=program_path,\
                    I=file_image_input,\
                    M=tflite_model,\
                    L=file_labels,\
                    height=input_height,\
                    width=input_width,\
                    mean=input_mean,\
                    std=input_std)

                # Log the command that will be executed.
                logger.info("Running command to label the image: {C}".format(C=cmd_label_image))

                # Execute the image classification program to label the image.
                result = ""
                if dry_run is True:
                    print(cmd_label_image)

                else:
                    # Create a subprocess to execute the image classification program.
                    process = subprocess.Popen(cmd_label_image, stdout=subprocess.PIPE)

                    # Get program stdout.
                    predictions = (process.communicate()[0]).decode("utf-8")

                    # Get program return code.
                    return_code = process.returncode

                    # Fetch image classification result if the image classification program doesn't return an error code.
                    if return_code == 0:
                        try:
                            # The program's stdout is prediction result as a JSON object string.
                            predictions_dict = json.loads(predictions)

                            # Log results.
                            logger.info("Model prediction results: " + predictions)

                            # Get label with highest prediction confidence.
                            label = max(predictions_dict.items(), key=operator.itemgetter(1))[0]
                            
                            # If the image classification is not greater or equal to a certain threshold then discard it.
                            if predictions_dict[label] < confidence_threshold:
                                logger.info("Insufficient prediction confidence level to label the image (the threshold is currently set to " + confidence_threshold + ").")
                            
                            else:
                                # Log highest confidence prediction.
                                logger.info("Labeling the image as '" + label + "'.")

                                # Check if the classified image should be ditched or kept based on what is set in the config.ini file.
                                if label not in labels_keep:
                                    logger.info("Ditching the image.")

                                else:
                                    # This image has been classified with a label of interest.
                                    # Keep the image but only the types as per what is configured in the the config.ini file.
                                    logger.info("Keeping the image.")
                                    
                                    # Remove the raw image file if it is not flagged to be kept.
                                    if not keep_raw:
                                        # FIXME: Consider using os.remove(file_raw). Would have to set file_raw first.
                                        cmd_remove_raw_image = 'rm ' + base_path + '/*.ims_rgb'
                                        os.system(cmd_remove_raw_image)

                                    # Remove the png image file if it is not flagged to be kept.
                                    if not keep_png:
                                        # FIXME: Consider using os.remove(file_png). The file_png variable already exists.
                                        cmd_remove_png_image = 'rm ' + base_path + '/*.png'
                                        os.system(cmd_remove_png_image)

                                    # Remove the jpeg image that as used as an input for the image classification program.
                                    # FIXME: Consider using os.remove(file_image_input). The file_image_input variable already exists.
                                    cmd_remove_input_image = 'rm ' + base_path + '/*_input.jpeg'
                                    os.system(cmd_remove_input_image)

                                    # Create a label directory in the experiment's toGround directory.
                                    # This is where the images will be moved to and how we categorize images based on their predicted labels.
                                    toGround_label_dir = toGround_path + '/' + label
                                    if not os.path.exists(toGround_label_dir):
                                        os.makedirs(toGround_label_dir)
                                    
                                    # Create the command to move the images to the experiment's toGround's label folder
                                    cmd_move_images = 'mv *.png *.ims_rgb *_thumbnail.jpeg {G}/'.format(G=toGround_label_dir)

                                    # Move the image to the experiment's toGround folder.
                                    os.system(cmd_move_images)

                            # Remove all images that have been leftover in the experiment's root directory.
                            # This should only matter for an image that was not labelled as an image of interest,
                            # in which case we are removing all images created during the image classification process.
                            remove_images(logger)

                        except:
                            # Log the exception.
                            logger.error("Failed to load predictions json: " + predictions)
                            logger.exception("message")

                    else: 
                        # Log error code if image classification program returned and error code.
                        logger.error("The image classification program returned error code " + str(return_code))

        except:
            # In case of exception just log the stack trace and proceed to the next image acquisition iteration.
            logger.exception("message")

        # Wait the configured sleep time before proceeding to the next image acquisition and labeling.
        logger.info("Wait {T} seconds...".format(T=gen_interval))
        time.sleep(gen_interval)
        
        # Increment image acquisition labeling counter.
        counter = counter + 1

        # Keep looping until the target iteration count is reached.
        if counter >= gen_number:
            done = True
        else:
            done = False

    # We have exited the image acquisition loop.
    # This means that we have finished acquiring images. 
    # It's time to tar the thumbnail images and the log file for downlinking.

    # Don't include the log files if we don't want to.
    if not log_keep:
        os.system('rm {L}/*.log'.format(L=log_path))

    # Use tar to package thumbnails and log files into the filestore's toGround folder.
    os.system('tar -czf {FSG}/opssat_hdcam_exp{expID}_{D}.tar.gz {G}/**/*.jpeg {L}/*.log --remove-files'.format(\
        FSG=filstore_toGroud,\
        expID=exp_id,\
        D=start_time.strftime("%Y%m%d_%H%M%S"),\
        G=toGround_path,\
        L=log_path))


def setup_logger(name, log_file, formatter, level=logging.INFO):
    """Setup the logger."""
    
    # Init handlers.
    fileHandler = logging.FileHandler(log_file)
    streamHandler = logging.StreamHandler()

    # Set formatters.
    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)

    # Init logger.
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)

    # Return the logger.
    return logger


if __name__ == '__main__':
    """Run the main program function after initializing the logger and timer."""

    # Init and configure the logger.
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    logging.Formatter.converter = time.gmtime
    start_time = datetime.datetime.utcnow()
    logfile = log_path + '/opssat_smartcamluvsu_{D}.log'.format(D=datetime.datetime.strftime(start_time, "%Y%m%d_%H%M%S"))
    smartcamluvsu_logger = setup_logger('smartcamluvsu_logger', log_path + '/opssat_smartcamluvsu_{D}.log'.format(D=start_time.strftime("%Y%m%d_%H%M%S")), formatter, level=logging.INFO)

    # Start the app.
    main(start_time, smartcamluvsu_logger, logfile)