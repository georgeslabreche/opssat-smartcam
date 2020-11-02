#!/usr/bin/python3

import platform
import os
import configparser
import logging
import time
import datetime
import json
import glob

__author__ = 'Georges Labreche, Georges.Labreche@esa.int'

# Select production or development environment dev path based on platform name.
base_path = '/home/root/georges/apps/SmartCamLuvsU/home/exp1000' if platform.node() == 'sepp' else '/home/georges/dev/SmartCamLuvsU/home/exp1000'

# Config path.
config_file = base_path + '/config.sepp.ini' if platform.node() == 'sepp' else 'config.dev.ini'

# Log path.
log_path = base_path + '/logs'

# img_path

# Image classifier program path.
program_path = base_path + '/bin/tensorflow/lite/c/image_classifier'

def cleanup(logger):
    ''' Delete image files that may remain in the experiment's home directory.
        These files could exist due to an unhandled error during a previous run.
    '''

    # Count the number of files deleted.
    delete_count = 0

    # Loop through all the file types to delete.
    for ext in ["ims_rgb", "png", "jpeg"]:

        # Get a list of all the file paths that ends with .txt from in specified directory
        img_files = glob.glob(base_path + "/*." + ext)

        for f in img_files:
            try:
                # TODO: Uncomment remove.
                #os.remove(f)
                delete_count = delete_count + 1
                logger.info("Deleted image file during cleanup: " + f)

            except:
                logger.error("Error deleting image file during cleanup: " + f)
                return -1

    return delete_count


def main(startTime, logger, logfile):
    ''' Run the experiment.
    '''

    # Init the config parser, read the config file.
    config = configparser.ConfigParser()
    config.read(config_file)

    # Determine if this is a dry run or not
    dry_run = config.getboolean('conf', 'dry_run')

    # Fetch model config values.
    tflite_model = config.get('model', 'tflite_model')
    input_size_x = config.get('model', 'input_size_x')
    input_size_y = config.get('model', 'input_size_y')

    # Fetch image acquisition parameters.
    gen_interval = config.getint('gen', 'gen_interval')
    
    gen_number = config.getint('gen', 'gen_number')
    if gen_number <= 0:
        gen_number = 1

    gen_exposure = config.getint('gen', 'gen_exposure')
    if gen_number <= 1:
        gen_exposure = 2

    gen_gains = json.loads(config.get('gen','gen_gains'))
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

            # Cleanup any files that may have been left over from the previous loop
            if cleanup(logger) < 0:
                return -1
            
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
                        X=input_size_x,\
                        Y=input_size_y,\
                        O=file_image_input)

                # Log the command that will be executed.
                logger.info("Running command to create input: {C}".format(C=cmd_create_input_image))

                # Run the command to create the image input file for the image classification program.
                os.system(cmd_create_input_image) if not dry_run else print(cmd_create_input_image)

                # Check that the image input exists.
                if not os.path.isfile(file_image_input):
                    logger.error(\
                        "Failed to generate {X}x{Y} image input for the image classification model.".format(\
                            X=input_size_x,
                            Y=input_size_y))

                    skip = True

            # If we have successffully create the input images then feed it into the iamge classification program.
            if not skip:

                cmd_label_image = program_path + ' ' + file_image_input + ' ' + tflite_model

                # Log the command that will be executed.
                logger.info("Running command to label the image: {C}".format(C=cmd_label_image))

                # Execute the image classification program to label the image.
                os.system(cmd_label_image) if not dry_run else print(cmd_create_input_image)
                #os.system(cmd_label_image)

                # TODO: 
                # 1. If image is of interest then move to experiment's toGround folder.
                # 2. If not of interest, remove!
                

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


    # TODO:
    # 1. tar the thumbnails in the experiment's toGround folder and move them to filestore's toGround folder (delete original thumbnails)
    # 2. Also tar the log files.


def setup_logger(name, log_file, formatter, level=logging.INFO):
    ''' Setup the logger.
    '''
    
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
    ''' Run the main program function after initializing the logger and timer.
    '''

    # Init and configure the logger.
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    logging.Formatter.converter = time.gmtime
    start_time = datetime.datetime.utcnow()
    logfile = log_path + '/opssat_smartcamluvsu_{D}.log'.format(D=datetime.datetime.strftime(start_time, "%Y%m%d_%H%M%S"))
    smartcamluvsu_logger = setup_logger('smartcamluvsu_logger', log_path + '/opssat_smartcamluvsu_{D}.log'.format(D=start_time.strftime("%Y%m%d_%H%M%S")), formatter, level=logging.INFO)

    # Start the app.
    main(start_time, smartcamluvsu_logger, logfile)