#!/usr/bin/python3

import platform
import os
import configparser
import logging
import time
import datetime

__author__ = 'Georges Labreche, Georges.Labreche@esa.int'

# Select production or development environment dev path based on platform name.
base_path = '/home/georges/apps/SmartCamLuvsU' if platform.node() == 'sepp' else '/home/georges/dev/SmartCamLuvsU/home/exp1000'

# Config path.
config_file = base_path + '/config.sepp.ini' if platform.node() == 'sepp' else 'config.dev.ini'

# Log path.
log_path = base_path + '/logs'

# Image classifier program path.
program_path = base_path + '/bin/tensorflow/lite/c/image_classifier'

def main(startTime, logger, logfile):

    # Init the config parser, read the config file.
    config = configparser.ConfigParser()
    config.read(config_file)

    # Fetch config values.
    img_dirs = config.get('conf', 'img_dir')
    tflite_model = config.get('conf', 'tflite_model')

    # Run image classification program for every image in the image directory.
    img_files = os.listdir(img_dirs)
    for f in img_files:
        program_command = program_path + ' ' + img_dirs + '/' + f + ' ' + tflite_model
        #print(program_command)

        # Execute the classification program.
        os.system(program_command)


def setup_logger(name, log_file, formatter, level=logging.INFO):
    
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

    # Init and configure the logger.
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    logging.Formatter.converter = time.gmtime
    start_time = datetime.datetime.utcnow()
    logfile = log_path + '/opssat_smartcamluvsu_{D}.log'.format(D=datetime.datetime.strftime(start_time, "%Y%m%d_%H%M%S"))
    smartcamluvsu_logger = setup_logger('smartcamluvsu_logger', log_path + '/opssat_smartcamluvsu_{D}.log'.format(D=start_time.strftime("%Y%m%d_%H%M%S")), formatter, level=logging.INFO)

    # Start the app.
    main(start_time, smartcamluvsu_logger, logfile)