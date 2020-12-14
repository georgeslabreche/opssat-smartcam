#!/usr/bin/python3

import os
import subprocess
import glob
import configparser
import logging
import time
import datetime
import json
import operator
import ntpath
import re
import csv
import ephem
from ephem import degree
from pathlib import Path

__author__ = 'Georges Labreche, Georges.Labreche@esa.int'

# The experiment id number.
EXP_ID = 1000

# The experiment's base path.
BASE_PATH = '/home/exp' + str(EXP_ID)

# The experiment's config file path.
CONFIG_FILE = BASE_PATH + '/config.ini'

# The experiment's toGround folder path.
TOGROUND_PATH = BASE_PATH + '/toGround'

# The filestore's toGround folder path.
FILESTORE_TOGROUND_PATH = '/home/root/esoc-apps/fms/filestore/toGround'

# Image classifier program file path.
IMAGE_CLASSIFIER_BIN_PATH = BASE_PATH + '/bin/tensorflow/lite/c/image_classifier'

# The fapce compression binary file path.
FAPEC_BIN_PATH = '/home/exp100/fapec'

# The supported compression types.
SUPPORTED_COMPRESSION_TYPES = ['fapec']

# The experiment start time.
START_TIME = datetime.datetime.utcnow()

# The experiment's log folder path.
LOG_PATH = BASE_PATH + '/logs'

# The experiment's log file path.
LOG_FILE = LOG_PATH + '/opssat_smartcam_{D}.log'.format(D=START_TIME.strftime("%Y%m%d_%H%M%S"))

# The experiment's metadata CSV file.
METADATA_CSV_FILE = LOG_PATH + '/opssat_smartcam_metadata_{D}.csv'.format(D=START_TIME.strftime("%Y%m%d_%H%M%S"))

# Image filename prefix.
IMG_FILENAME_PREFIX = "img_msec_"

# The logger.
logger = None

class AppConfig:

    def __init__(self):
        # Init the config parser, read the config file.
        self.config = configparser.ConfigParser()
        self.config.read(CONFIG_FILE)

        # Init the conf config section properties.
        self.init_conf_props()

        # Init the gen config section properties.
        self.init_gen_props()

        # Init the img config section properties.
        self.init_img_props()


    def init_conf_props(self):
        """Fetch general configuration parameters."""

        # Flag if logs should be downlinked even if no images are classified to be kept.
        self.downlink_log_if_no_images = self.config.getboolean('conf', 'downlink_log_if_no_images')

        # The first model to apply.
        self.entry_point_model = self.config.get('conf', 'entry_point_model')

        # The compression type to apply.
        self.raw_compression_type = self.config.get('conf', 'raw_compression_type')

        # The size in which the packaged raw images should be split.
        self.downlink_compressed_split = self.config.get('conf', 'downlink_compressed_split')

        # Flag if thumbnails should be downlinked.
        self.downlink_thumbnails = self.config.getboolean('conf', 'downlink_thumbnails')

        # Flag if the packaged raw images should be automatically moved to the filestore's toGround folder.
        # If yes, then the images will be downlinked during the next pass.
        self.downlink_compressed_raws = self.config.getboolean('conf', 'downlink_compressed_raws')

        # Flag to collect image metadata into a CSV file.
        self.collect_metadata = self.config.getboolean('conf', 'collect_metadata')

        # TLE file path.
        self.tle_path = self.config.get('conf', 'tle_path')

        # Size quote for the experiment's toGround folder.
        self.quota_toGround = self.config.getint('conf', 'quota_toGround')


    def init_model_props(self, model_name):
        """Fetch model configuration parameters."""

        # Get the config section name for the current model.
        model_cfg_section_name = 'model_' + model_name

        # Check that the model section exists in the configuration file before proceeding.
        if self.config.has_section(model_cfg_section_name) is False:
            return False

        # Fetch the model configuration properties.
        self.tflite_model = self.config.get(model_cfg_section_name, 'tflite_model')
        self.file_labels = self.config.get(model_cfg_section_name, 'labels')
        self.labels_keep = json.loads(self.config.get(model_cfg_section_name, 'labels_keep'))
        self.input_height = self.config.get(model_cfg_section_name, 'input_height')
        self.input_width = self.config.get(model_cfg_section_name, 'input_width')
        self.input_mean = self.config.get(model_cfg_section_name, 'input_mean')
        self.input_std = self.config.get(model_cfg_section_name, 'input_std')
        self.confidence_threshold = self.config.get(model_cfg_section_name, 'confidence_threshold')

        return True


    def init_compression_fapec_props(self):
        """Fetch fapec compression parameters."""

         # Get the config section name for fapec compression.
        fapec_cfg_section_name = 'compression_fapec'

        # Check that the model section exists in the configuration file before proceeding.
        if self.config.has_section(fapec_cfg_section_name) is False:
            return False
        
        self.compression_fapec_chunk = self.config.get(fapec_cfg_section_name, 'chunk')
        self.compression_fapec_threads = self.config.getint(fapec_cfg_section_name, 'threads')
        self.compression_fapec_dtype = self.config.getint(fapec_cfg_section_name, 'dtype')
        self.compression_fapec_band = self.config.getint(fapec_cfg_section_name, 'band')
        self.compression_fapec_losses = self.config.get(fapec_cfg_section_name, 'losses')
        self.compression_fapec_meaningful_bits = self.config.getint(fapec_cfg_section_name, 'meaningful_bits')
        self.compression_fapec_lev = self.config.getint(fapec_cfg_section_name, 'lev')

        return True


    def init_gen_props(self):
        """Fetch image acquisition parameters."""

        self.gen_interval = self.config.getint('gen', 'gen_interval')

        self.gen_interval_throttle = self.config.getint('gen', 'gen_interval_throttle')

        self.gen_number = self.config.getint('gen', 'gen_number')
        if self.gen_number <= 0:
           self. gen_number = 1

        self.gen_exposure = self.config.getint('gen', 'gen_exposure')
        if self.gen_exposure <= 1:
            self.gen_exposure = 2

        self.gen_gains = json.loads(self.config.get('gen', 'gen_gains'))
        if self.gen_gains[0] >= 255:
            self.gen_gains[0] = 255

        if self.gen_gains[1] >= 255:
            self.gen_gains[1] = 255

        if self.gen_gains[2] >= 255:
            self.gen_gains[2] = 255


    def init_img_props(self):
        """Fetch image parameters."""

        # Fetch image file retention parameters.
        self.raw_keep = self.config.getboolean('img', 'raw_keep')
        self.png_keep = self.config.getboolean('img', 'png_keep')

        # Fetch jpeg thumbnail image processing parameters.
        self.jpeg_scaling = self.config.getfloat('jpeg', 'jpeg_scaling')
        if self.jpeg_scaling >= 1.0 or self.jpeg_scaling <= 0:
            self.jpeg_scaling = 0.5  

        self.jpeg_quality = self.config.getint('jpeg', 'jpeg_quality')
        if self.jpeg_quality >= 100 or self.jpeg_quality <= 0:
            self.jpeg_quality = 90

        self.jpeg_processing = self.config.get('jpeg', 'jpeg_processing')
        if self.jpeg_processing != 'pnmnorm' and self.jpeg_processing != 'pnmhisteq':
            self.jpeg_processing = 'none'


class ImageMetaData:

    tle = None

    FIELD_NAMES = [
        'filename',     # Filename without extension.
        'label',        # Label applied to the image by the image classifier.
        'confidence',   # Confidence level of th the applied label.
        'keep',         # Whether the image was kept or not.
        'gain_r',       # Camera gain setting for red channel.
        'gain_g',       # Camera gain setting for green channel.
        'gain_b',       # Camera gain setting for blue channel.
        'exposure',     # Camera exposure setting (ms).
        'acq_ts',       # Image acquisition timestamp.
        'acq_dt',       # Image acquisition datetime.
        'ref_dt',       # Reference epoch.
        'tle_age',      # TLE age (days).
        'lat',          # Latitude (deg).
        'lng',          # Longitude (deg).
        'h',            # Geocentric height above sea level (m).
        'tle_ref_line1',# Reference TLE line 1.
        'tle_ref_line2' # Reference TLE line 2.
    ]


    def __init__(self, tle_path, gains, exposure):
        
        # The TLE.
        tle = None

        # Lines 1 and 2 from the TLE file.
        tle_line1 = None
        tle_line2 = None

        # Read lines from TLE file.
        try:
            lines = None

            with open(tle_path, 'r') as tle_file:
                lines = tle_file.readlines()

                if lines is not None:
                
                    if len(lines) == 2:
                        self.tle_line1 = lines[0].rstrip()
                        self.tle_line2 = lines[1].rstrip()

                        # Read the TLE.
                        self.tle = ephem.readtle("OPS-SAT", self.tle_line1, self.tle_line2)

                    elif len(lines) >= 3:
                        self.tle_line1 = lines[1].rstrip()
                        self.tle_line2 = lines[2].rstrip()

                        # Read the TLE.
                        self.tle = ephem.readtle(lines[0], self.tle_line1, self.tle_line2)

        except:
            self.tle = None
            logger.error("Failed to read TLE file: " + tle_path)

        self.gains = gains
        self.exposure = exposure

        # The list that will contain metadata dictionary entries.
        self.metadata_list = []


    def collect_metadata(self, filename_png, label, confidence, keep):
        """Collect metadata for the image acquired at the given timestamp."""

        # Track if tle compuation is successful or not.
        # If not successful then fallback to minimum metadata collection that does not depend on TLE.
        tle_compute_success = False

        # Image acquisition timestamp.
        timestamp = None
        
        # The dictionary that will contain the image's computed metadata.
        metadata = {}

        # The filename without the path and without the extension
        filename = filename_png.replace(BASE_PATH + "/", "").replace(".png", "")

        try:
            # Extract timestamp from filename.
            timestamp = int(re.match(IMG_FILENAME_PREFIX + "(\d+)_\d+", filename).group(1))
        
        except:
            logger.exception("Failed to extract timestamp from the image filename.")

        if timestamp is not None and self.tle is not None:
            try: 

                # Image acquisition datetime.
                d = datetime.datetime.utcfromtimestamp(timestamp / 1000.0)

                # Image acquisition epheme datetime object.
                d_ephem = ephem.Date(d)

                # Compute based on reference TLE and acquisition datetime.
                self.tle.compute(d_ephem)

                # Build the metadata dictionary for the acquired image.
                metadata = {
                    'filename': filename,                 # Filename without extension.
                    'label': label,                       # Label applied to the image by the image classifier.
                    'confidence': confidence,             # Confidence level of th the applied label.
                    'keep': keep,                         # Whether the image was kept or not.
                    'gain_r': self.gains[0],              # Camera gain setting for red channel.
                    'gain_g': self.gains[1],              # Camera gain setting for green channel.
                    'gain_b': self.gains[2],              # Camera gain setting for blue channel.
                    'exposure': self.exposure,            # Camera exposure setting (ms).
                    'acq_ts': timestamp,                  # Image acquisition timestamp.
                    'acq_dt': str(d_ephem),               # Image acquisition datetime.
                    'ref_dt': str(self.tle._epoch),       # Reference epoch.
                    'tle_age': d_ephem - self.tle._epoch, # TLE age (days).
                    'lat': self.tle.sublat / degree,      # Latitude (deg).
                    'lng': self.tle.sublong / degree,     # Longitude (deg).
                    'h': self.tle.elevation,              # Geocentric height above sea level (m).
                    'tle_ref_line1': self.tle_line1,      # Reference TLE line 1.
                    'tle_ref_line2': self.tle_line2       # Rererence TLE line 2.
                }

                # TLE computer success.
                tle_compute_success = True

            except:
                # Log exception.
                logger.exception("Failed TLE computation with image timestamp: " + str(timestamp))


        # Computing some metadata with the available TLE failed.
        # Try collecting basic metadata that does not depend on TLE.
        if not tle_compute_success:

            try:
                # Limited metadata in case of TLE error.
                metadata = {
                    'filename': filename,                 # Filename without extension.
                    'label': label,                       # Label applied to the image by the image classifier.
                    'confidence': confidence,             # Confidence level of th the applied label.
                    'keep': keep,                         # Whether the image was kept or not.
                    'gain_r': self.gains[0],              # Camera gain setting for red channel.
                    'gain_g': self.gains[1],              # Camera gain setting for green channel.
                    'gain_b': self.gains[2],              # Camera gain setting for blue channel.
                    'exposure': self.exposure,            # Camera exposure setting (ms).
                    'acq_ts': timestamp,                  # Image acquisition timestamp.
                    'tle_ref_line1': self.tle_line1,      # Reference TLE line 1.
                    'tle_ref_line2': self.tle_line2       # Rererence TLE line 2.
                }

            except:
                # No reason for this to occur but here nevertheless, just in case.
                logger.exception("Failed to collect metadata.")

        # Append metadata to the dictionary. 
        # Will be written into a CSV file at the end of the image acquisition loop.
        if len(metadata) > 0:
            self.metadata_list.append(metadata)
        

    def write_metadata(self, csv_filename):
        """Write collected metadata into a CSV file."""

        if len(self.metadata_list) > 0:

            # Open CSV file and start writing image metadata row for each image acquired.
            with open(csv_filename, 'w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.FIELD_NAMES)

                # Write header.
                writer.writeheader()

                # Write image metadata row.
                for metadata in self.metadata_list:
                    writer.writerow(metadata)

        else:
            logger.info("No metadata data was collected.")


class Fapec:

    bin_path = FAPEC_BIN_PATH

    def __init__(self, chunk, threads, dtype, band, losses, meaningful_bits, lev):
        """Initialize the Fapec compression class."""

        self.chunk = chunk
        self.threads = threads
        self.dtype = dtype
        self.band = band
        self.losses = losses
        self.meaningful_bits = meaningful_bits
        self.lev = lev


    def compress(self, src, dst):
        """Compress the given file(s)."""

        #TODO: Check if lev should not be included when set to 0.

        # The fapec compression command with all parameters.
        cmd_compress = '{BIN} -q -chunk {C} -mt {T} -dtype {DT} -cillic 2048 1944 {B} {L} {MB} 4 {LEV} -ow -o {DST} {SRC} >> {LOG} 2>&1'.format(\
            BIN=self.bin_path,\
            C=self.chunk,\
            T=self.threads,\
            DT=self.dtype,\
            B=self.band,\
            L=self.losses,\
            MB=self.meaningful_bits,\
            LEV='-lev ' + str(self.lev) if self.lev > 0 else '',\
            SRC=src,\
            DST=dst,\
            LOG=LOG_FILE)

        # Log compression command that will be executed.
        logger.info("Running command to compress image: {C}".format(C=cmd_compress))

        # Apply the compression.
        os.system(cmd_compress)


class Utils:

    def cleanup(self):
        """Delete files created on the project's root directory while processing the acquired image.

        These files could exist due to an unhandled error during a previous run so as a precaution 
        we also run this function prior to image acquisition.
        """

        # Count the number of files deleted.
        delete_count = 0

        # Loop through all the file types to delete.
        for ext in ['ims_rgb', 'png', 'jpeg', 'tar', 'tar.gz']:

            # Get a list of all the file paths that ends with .txt from in specified directory
            img_files = glob.glob(BASE_PATH + "/*." + ext)

            for f in img_files:
                try:
                    os.remove(f)
                    delete_count = delete_count + 1
                    logger.info("Removed file: " + f)

                except:
                    logger.error("Error removing file: " + f)
                    return -1

        return delete_count


    def get_image_keep_status_and_next_model(self, applied_label, labels_keep):
        """Determine whether or not the image should be kept as well as the next model that should be applied."""
        
        # Check if the labeled image should be ditched or kept based on what is set in the config.ini file.
        for lbl_k in labels_keep:

            # The label value can be a pair of values represented as a colon seperated string.
            #
            # If it is a pair:
            #   - the first value represents which labelled image to keep.
            #   - the second value represent the follow up model to apply to the kept image.
            #
            # If it is not a pair then the value only represents which labeled image to keep. There is no follow-up model.
            label_split = lbl_k.split(':')

            # If the applied label is marked for keeping
            if applied_label == label_split[0]:

                # Determine if there is a follow up model to apply to this image.
                if len(label_split) == 1:
                    # No follow up model to apply after this current model.
                    next_model = None

                elif len(label_split) == 2:
                    # There is a follow up model to apply after this current model.
                    next_model = label_split[1]

                else:
                    # Multiple follw up models were defined.
                    # Log a warning that this is not ssupported.
                    # TODO: Support this case of "model branching". Look into approaching this with a node graph.
                    logging.warning("Branching to multiple follow up models is currently unsupported. Selecting the first next model listed.")

                    # Only follow up with the first follow up model that is listed.
                    next_model = label_split[1]

                # Return that the image is to be kept and the name of the next model to apply on the kept image.
                return True, next_model

        # Don't keep the image and, of course, no next model to follow up with.
        return False, None


    def move_images_for_keeping(self, raw_keep, png_keep, applied_label):
        """Move the images to keep into to experiment's toGround folder."""

        # Remove the raw image file if it is not flagged to be kept.
        if not raw_keep:
            cmd_remove_raw_image = 'rm ' + BASE_PATH + '/*.ims_rgb'
            os.system(cmd_remove_raw_image)

        # Remove the png image file if it is not flagged to be kept.
        if not png_keep:
            cmd_remove_png_image = 'rm ' + BASE_PATH + '/*.png'
            os.system(cmd_remove_png_image)

        # Remove the jpeg image that was used as an input for the image classification program.
        cmd_remove_input_image = 'rm ' + BASE_PATH + '/*_input.jpeg'
        os.system(cmd_remove_input_image)

        # Create a label directory in the experiment's toGround directory.
        # This is where the images will be moved to and how we categorize images based on their predicted labels.
        toGround_label_dir = TOGROUND_PATH + '/' + applied_label
        if not os.path.exists(toGround_label_dir):
            os.makedirs(toGround_label_dir)
        
        # Create the command to move the images to the experiment's toGround's label folder
        cmd_move_images = 'mv *.png *.ims_rgb *_thumbnail.jpeg {G}/'.format(G=toGround_label_dir)

        # Move the image to the experiment's toGround folder.
        os.system(cmd_move_images)


    def package_files_for_downlinking(self, file_ext, downlink_log_if_no_images):
        """Package the files for downlinking."""
        try:

            # Don't use gzip if files are already a compression file type.
            # Check against a list of file types in case multiple compression types are supported.
            tar_options = '-cf' if file_ext in SUPPORTED_COMPRESSION_TYPES else '-czf'
            tar_ext = 'tar' if file_ext in SUPPORTED_COMPRESSION_TYPES else 'tar.gz' 

            # The destination tar file path for the packaged files.
            tar_path = '{TG}/opssat_smartcam_{FILE_EXT}_exp{expID}_{D}.{TAR_EXT}'.format(\
                    TG=TOGROUND_PATH,\
                    FILE_EXT=file_ext,\
                    expID=EXP_ID,\
                    D=START_TIME.strftime("%Y%m%d_%H%M%S"),\
                    TAR_EXT=tar_ext)

            # Count how many images were kept and moved to the experiment's toGround folder.
            image_count = len(list(Path(TOGROUND_PATH).rglob('*.' + file_ext)))

            # Count how many log files were produced.
            log_count = len(list(Path(LOG_PATH).rglob('*.log')))
            
            # Tar images(s) for downlink if at least 1 image was classified with a label of interest.
            if image_count > 0:

                # Log that we are tarring some images.
                logger.info("Tarring {T} file(s) for downlink.".format(T=image_count))

                # Use tar to package image and log files into the filestore's toGround folder.
                os.system('tar {TAR_O} {TAR_PATH} {G}/**/*.{FILE_EXT} {L}/*.log {L}/*.csv --remove-files'.format(\
                    TAR_O=tar_options,\
                    TAR_PATH=tar_path,\
                    G=TOGROUND_PATH,\
                    FILE_EXT=file_ext,\
                    L=LOG_PATH))

                # Return experiment toGround path to tar file.
                return tar_path

            elif downlink_log_if_no_images is True and log_count > 0:

                # Log that we are only tarring log files.
                logger.info("No image(s) kept but tarring logs for downlink.")

                # The destination tar file path for the packaged files.
                tar_path = '{TG}/opssat_smartcam_{FILE_EXT}_exp{expID}_{D}.{TAR_EXT}'.format(\
                    TG=TOGROUND_PATH,\
                    FILE_EXT='logs',\
                    expID=EXP_ID,\
                    D=START_TIME.strftime("%Y%m%d_%H%M%S"),\
                    TAR_EXT=tar_ext)

                # Use tar to package log files into the filestore's toGround folder.
                os.system('tar {TAR_O} {TAR_PATH} {L}/*.log {L}/*.csv --remove-files'.format(\
                    TAR_O=tar_options,\
                    TAR_PATH=tar_path,\
                    L=LOG_PATH))

                # Return experiment toGround path to tar file.
                return tar_path

            else:
                # No images and no logs. Unlikely.
                logger.info("No images(s) kept nor logs produced for downlink.")

                # Return None.
                return None

        except:
            # In case this happens, the image will be tarred at the end of the next experiment's run unless explicitely deleted.
            logger.exception("Failed to tar kept image for downlink (if any).")

            # Return None.
            return None


    def split_and_move_tar(self, tar_path, split_bytes):
        """Split packaged files for downlink and move the chunks to the filestore's toGround folder."""
        
        # Thumbnail packages can be large when acquiring a lot of images.
        # Split the tar file and save smaller chunks in filestore's toGround folder.
        cmd_split_tar = 'split -b {B} {T} {P}'.format(\
            B=split_bytes,\
            T=tar_path,\
            P=FILESTORE_TOGROUND_PATH + "/" + ntpath.basename(tar_path) + "_")

        # Move the split chunks of the tar package to filestore's toGround folder.
        os.system(cmd_split_tar)

        # Get the number of files that the tar file was split into, i.e. the number of chunks.
        chunk_counter = len(glob.glob1(FILESTORE_TOGROUND_PATH, ntpath.basename(tar_path) + "_*"))

        # If the split only resulted in 1 chunk this means that no split was required the begin with. Rename the file as an unsplit tar file.
        # FIXME: Don't split to begin with when this will be the case (compare the tar size to the split size prior to splitting)
        if chunk_counter == 1:

            cmd_rename_single_chunk = 'mv {S} {D}'.format(\
                S=FILESTORE_TOGROUND_PATH + "/" + ntpath.basename(tar_path) + "_aa",\
                D=FILESTORE_TOGROUND_PATH + "/" + ntpath.basename(tar_path))

            os.system(cmd_rename_single_chunk)

        # Delete the unsplit tar file in the experiment's toGround folder.
        os.system('rm {T}'.format(T=tar_path))


    def log_housekeeping_data(self):
        """Log some housekeeping data, i.e. the available disk space."""

        # Disk usage.
        df_output = subprocess.check_output(['df', '-h']).decode('utf-8')
        logger.info('Disk usage:\n' + df_output) 

class HDCamera:

    def __init__(self, gains, exposure):
        self.gains = gains
        self.exposure = exposure

    def acquire_image(self):
        """Acquire and image with the on-board camera."""

        # Build the image acquisition execution command string.
        cmd_image_acquisition = 'ims100_testapp -R {R} -G {G} -B {B} -c /dev/ttyACM0 -m /dev/sda -v 0 -n 1 -p -e {E} >> {L} 2>&1'.format(\
            R=self.gains[0],\
            G=self.gains[1],\
            B=self.gains[2],\
            E=self.exposure,\
            L=LOG_FILE)
        
        # Log the command that will be executed.
        logger.info("Running command to acquire an image: {C}".format(C=cmd_image_acquisition))

        # Run the image acquisition command. 
        os.system(cmd_image_acquisition)

        # Check that png file exists...
        png_files = glob.glob(BASE_PATH + "/*.png")
        
        # If the png file doesn't exist then skip this iteration.
        if len(png_files) != 1:
            logger.error("Failed to acquire an image from the camera.")
            return None
            
        else:
            file_png = png_files[0]
            
            # An error in running the ims100_test_app command could result in an empty image file.
            # Make sure this is not the case.
            if Path(file_png).stat().st_size > 0:
                logger.info("Acquired image: " + file_png)

            else:
                # Empty image file acquired. Log error and skip image classification.
                logger.error("Image acquired from the camera is an empty file (0 KB).")
                return None

        # Return error status
        return file_png


class ImageEditor:

    def create_thumbnail(self, png_src_filename, jpeg_dest_filename, jpeg_scaling, jpeg_quality, jpeg_processing):
        """Create a thumbnail image."""

        # Build the thumbnail creation command string.
        if jpeg_processing != 'none':
            cmd_create_thumbnail = 'pngtopam {F} | pamscale {S} | {P} | pnmtojpeg -quality {Q} > {O}'.format(\
                S=jpeg_scaling,\
                Q=jpeg_quality,\
                P=jpeg_processing,\
                F=png_src_filename,\
                O=jpeg_dest_filename)

        else:
            cmd_create_thumbnail = 'pngtopam {F} | pamscale {S} | pnmtojpeg -quality {Q} > {O}'.format(\
                S=jpeg_scaling,\
                Q=jpeg_quality,\
                F=png_src_filename,\
                O=jpeg_dest_filename)

        # Log the command that will be executed.
        logger.info("Running command to create thumbnail: {C}".format(C=cmd_create_thumbnail))

        # Run the thumbnail creation command.
        os.system(cmd_create_thumbnail)

        # Check that thumbnail exists.
        if not os.path.isfile(jpeg_dest_filename):
            logger.error("Failed to generate a thumbnail.")
            
            return False

        # An error in executing the pngtopan, pamscale, or pnmtpjpeg commands can produce an empty thumbnail file.
        if Path(jpeg_dest_filename).stat().st_size == 0:
            logger.error("Generated thumbnail is an empty file (0 KB).")
            
            return False

        return True


    def create_input_image(self, png_src_filename, jpeg_dest_filename, input_height, input_width, jpeg_scaling, jpeg_quality, jpeg_processing):
        """Create image file as an input to the image classifier."""

        # Build the command string to create the image input for the image classification program.
        # FIXME create input jpeg directly from the thumbnail jpeg instead of from the png. Maye require an ipk to install jpegtopnm.
        if jpeg_processing != 'none':
            cmd_create_input_image = 'pngtopam {F} | pamscale -xsize {X} -ysize {Y} | {P} | pnmtojpeg -quality {Q} > {O}'.format(\
                F=png_src_filename,\
                Y=input_height,\
                X=input_width,\
                P=jpeg_processing,\
                Q=jpeg_quality,\
                O=jpeg_dest_filename)

        else:
            cmd_create_input_image = 'pngtopam {F} | pamscale -xsize {X} -ysize {Y} | pnmtojpeg -quality {Q} > {O}'.format(\
                F=png_src_filename,\
                Y=input_height,\
                X=input_width,\
                Q=jpeg_quality,\
                O=jpeg_dest_filename)

        # Log the command that will be executed.
        logger.info("Running command to create input: {C}".format(C=cmd_create_input_image))

        # Run the command to create the image input file for the image classification program.
        os.system(cmd_create_input_image)

        # Check that the image input exists.
        if not os.path.isfile(file_image_input):
            logger.error(\
                "Failed to generate {X}x{Y} image input for the image classification model.".format(\
                    X=input_width,
                    Y=input_height))

            return False

        # An error in executing the pamscale command can produce an empty image input file.
        if Path(file_image_input).stat().st_size == 0:
            logger.error("Generated image input is an empty file (0 KB).")
            
            return False

        # Return success boolean value.
        return True


class ImageClassifier:
    
    def label_image(self, image_filename, model_tflite_filename, labels_filename, image_height, image_width, image_mean, image_std):
        """Label an image using the image classifier with the given model and labels files."""

        try:
            # Build the image labeling command.
            cmd_label_image = '{P} {I} {M} {L} {height} {width} {mean} {std}'.format(\
                P=IMAGE_CLASSIFIER_BIN_PATH,\
                I=image_filename,\
                M=model_tflite_filename,\
                L=labels_filename,\
                height=image_height,\
                width=image_width,\
                mean=image_mean,\
                std=image_std)

            # Log the command that will be executed.
            logger.info("Running command to label the image: {C}".format(C=cmd_label_image))

            # Create a subprocess to execute the image classification program.
            process = subprocess.Popen(cmd_label_image, stdout=subprocess.PIPE, shell=True)

            # Get program stdout.
            predictions = (process.communicate()[0]).decode("utf-8")

            return_code = process.returncode 

            # Get program return code.
            if return_code == 0:
                # Log results.
                logger.info("Model prediction results: " + predictions)

                # The program's stdout is prediction result as a JSON object string.
                return json.loads(predictions)
            
            else: 
                # Log error code if image classification program returned and error code.
                logger.error("The image classification program returned error code {E}.".format(E=str(return_code)))

        except:
            # Log the exception.
            logger.exception("An error was thrown while attempting to run the image classification program.")

        return None


def run_experiment():
    """Run the experiment."""

    # The config parser.
    cfg = AppConfig()

    utils = Utils()
    camera = HDCamera(cfg.gen_gains, cfg.gen_exposure)
    img_editor = ImageEditor()
    img_classifier = ImageClassifier()
    img_metadata = ImageMetaData(cfg.tle_path, cfg.gen_gains, cfg.gen_exposure)

    # Flag and counter to keep track.
    done = False
    counter = 0

    # Check if the experiment's toGround folder is below a configured quota before proceeding with the experiment.
    try:
        toGround_size = int(subprocess.check_output(['du', '-s', TOGROUND_PATH]).decode('utf-8').split()[0])
        done = True if toGround_size >= cfg.quota_toGround else False

        if done:
            logger.info("Exiting: the experiment's toGround folder is greater than the configured quota: {TG} KB > {Q} KB.".format(\
                TG=toGround_size,\
                Q=cfg.quota_toGround))

    except:
        logger.exception("Exiting: failed to check disk space use of the experiment's toGround folder.")
        done = False
    
    # Instanciate a compressor object if a compression algorithm was specified and configured in the config.ini.
    raw_compressor = None

    # Proceed with instanciate the compressor object in case the is not marked to stop.
    if not done:

        # Raw image file compression will only be applied if we enable compressed raw downlinking.
        if cfg.downlink_compressed_raws and cfg.raw_compression_type == 'fapec' and cfg.init_compression_fapec_props() is True:

            # Instanciate compression object that will be used to compress the raw image files.
            raw_compressor = Fapec(cfg.compression_fapec_chunk,\
                cfg.compression_fapec_threads,\
                cfg.compression_fapec_dtype,\
                cfg.compression_fapec_band,\
                cfg.compression_fapec_losses,\
                cfg.compression_fapec_meaningful_bits,\
                cfg.compression_fapec_lev)

            logger.info("Raw image file compression enabled: " + cfg.raw_compression_type + ".")

        else:
            # No compression will be applied to the raw image files.
            logger.info("Raw image file compression disabled.")

    # Default immage acquisition interval. Can be throttled when an acquired image is labeled to keep.
    image_acquisition_period = cfg.gen_interval

    # Image acquisition loop.
    while not done:

        try:

            # Flag indicating if we should skip the image acquisition and labeling process in case of an encountered error.
            success = True

            # Cleanup any files that may have been left over from a previous run that may have terminated ungracefully.
            if utils.cleanup() < 0:
                success = False

            # If experiment's root directory is clean, i.e. no images left over from a previous image acquisition, then acquire a new image.
            if success:
                # Acquire the image.
                file_png = camera.acquire_image()

                # Check if image acquisition was OK.
                success = True if file_png is not None else False
            
            # If we have successfully acquired a png file then create the  jpeg thumbnail if we want to downlink thumbnails.
            if success and cfg.downlink_thumbnails:
                # The thumbnail filename.
                file_thumbnail = file_png.replace(".png", "_thumbnail.jpeg")
                
                # Create the thumbnail.
                success = img_editor.create_thumbnail(file_png, file_thumbnail, cfg.jpeg_scaling, cfg.jpeg_quality, cfg.jpeg_processing)

            # Proceed if we have successfully create the thumbnail image.
            if success:

                # Set the first image classification model to apply.
                next_model = cfg.entry_point_model

                # Keep applying follow up models to the kept image as long as images are labeled to be kept and follow up models are defined.
                while next_model is not None:

                    # Assuming the image will not be kept until we get the final result from the last model in the pipeline.
                    keep_image = False

                    # Init the model configuration properties for the current modell
                    success = cfg.init_model_props(next_model)

                    # Check that the model section exists in the configuration file before proceeding.
                    if not success:
                        logger.error("Skipping the '{M}' model: it is not defined in the config.ini file.".format(M=next_model))
                        break

                    else:
                        # Logging which model in the pipeline is being used to classify the image
                        logger.info("Labeling the image using the '{M}' model.".format(M=next_model))

                    # File name of the image file that will be used as the input image to feed the image classification model.
                    file_image_input = file_png.replace(".png", "_input.jpeg")

                    # Create the image that will be used as the input for the neural network image classification model.
                    success = image_editor.create_input_image(\
                        file_png, file_image_input,\
                        cfg.input_height, cfg.input_width,\
                        cfg.jpeg_scaling, cfg.jpeg_quality, cfg.jpeg_processing)

                    # Input image for the model was successfully created, proceed with running the image classification program.
                    if success:
                        # Label the image.
                        predictions_dict = img_classifier.label_image(\
                            file_image_input, cfg.tflite_model, cfg.file_labels,\
                            cfg.input_height, cfg.input_width, cfg.input_mean, cfg.input_std)

                        # Break out of the loop if the image classification program returns an error.
                        if predictions_dict is None:

                            # Break out of the loop.
                            break

                        # Fetch image classification result if the image classification program doesn't return an error code.
                        elif predictions_dict:

                            # Get label with highest prediction confidence.
                            applied_label = max(predictions_dict.items(), key=operator.itemgetter(1))[0]
                            
                            # Get the confidence value of the label with the higher confidence.
                            applied_label_confidence = float(predictions_dict[applied_label])

                            # If the image classification is not greater or equal to a certain threshold then discard it.
                            if applied_label_confidence < float(cfg.confidence_threshold):
                                logger.info("Insufficient prediction confidence level to label the image (the threshold is currently set to " + cfg.confidence_threshold + ").")

                                # Break out of the loop if the prediction confidence is not high enough and we cannot proceed in labeling the image.
                                break
                            
                            else:
                                # Log highest confidence prediction.
                                logger.info("Labeling the image as '" + applied_label + "'.")

                                # Determine if we are keeping the image and if we are applying another classification model to it.
                                # If next_model is not None then proceed to another iteration of this model pipeline loop.
                                keep_image, next_model = utils.get_image_keep_status_and_next_model(applied_label, cfg.labels_keep)


                # We have exited the model pipeline loop.
                
                # Collect image metadata. Even for images that will not be kept.
                if predictions_dict is not None and cfg.collect_metadata:
                    img_metadata.collect_metadata(file_png, applied_label, applied_label_confidence, keep_image)

                # Remove the image if it is not labeled for keeping.
                if not keep_image:
                    # Log image removal.
                    logger.info("Ditching the image.")

                    # The acquired image is not of interest: fall back the default image acquisition frequency.
                    image_acquisition_period = cfg.gen_interval

                    # Remove image.
                    utils.cleanup()
                
                # Move the image to the experiment's toGround folder if we have gone through all the
                # models in the pipeline and still have an image that is labeled to keep for downlinking.
                else:

                    # The current image has been classified with a label of interest.
                    # Keep the image but only the types as per what is configured in the the config.ini file.
                    logger.info("Keeping the image.")

                    # Compress raw image if configured to do so.
                    if raw_compressor is not None:

                        # Log message to indicate compression.
                        logger.info("Compressing the raw image.")
                        
                        # Source and destination file paths for raw image file compression.
                        file_raw_image = file_png.replace(".png", ".ims_rgb")
                        file_raw_image_compressed = TOGROUND_PATH + "/" + applied_label + "/" + ntpath.basename(file_png).replace(".png", "." + cfg.raw_compression_type)

                        # Create a label directory in the experiment's toGround directory.
                        # This is where the compressed raw image file will be moved to and how we categorize images based on their predicted labels.
                        toGround_label_dir = TOGROUND_PATH + '/' + applied_label
                        if not os.path.exists(toGround_label_dir):
                            os.makedirs(toGround_label_dir)

                        # Compress the raw image file.
                        raw_compressor.compress(file_raw_image, file_raw_image_compressed)

                    # Move the images for keeping.
                    utils.move_images_for_keeping(cfg.raw_keep, cfg.png_keep, applied_label)

                    # An image of interest has been acquired: throttle image acquisition frequency.
                    image_acquisition_period =  cfg.gen_interval_throttle

        except:
            # In case of exception just log the stack trace and proceed to the next image acquisition iteration.
            logger.exception("Failed to acquire and classify image.")

        # Error handling here to not risk an unlikely infinite loop.
        try:

            # Increment image acquisition labeling counter.
            counter = counter + 1

            # Wait the configured sleep time before proceeding to the next image acquisition and labeling.
            if counter < cfg.gen_number:
                logger.info("Wait {T} seconds...".format(T=image_acquisition_period))
                time.sleep(image_acquisition_period)
            else:
                logger.info("Image acquisition loop completed.")
            
            # Keep looping until the target iteration count is reached.
            if counter >= cfg.gen_number:
                done = True
            else:
                done = False

        except:
            # An unlikely exception is preventing the loop counter to increment.
            # Log the exception and exit the loop.
            logger.exception("An unlikely failure occured while waiting for the next image acquisition.")
            done = True

    # We have exited the image acquisition and labeling loop.
    # This means that we have finished labeling the acquired images. 

    # Log some housekeeping data.
    utils.log_housekeeping_data()

    # Write metadata CSV file.
    if cfg.collect_metadata:
        img_metadata.write_metadata(METADATA_CSV_FILE)

    # Tar the images and the log files for downlinking.

    # Package thumbnails for downlinking.
    if cfg.downlink_thumbnails:
        tar_path = utils.package_files_for_downlinking("jpeg", cfg.downlink_log_if_no_images)

        if tar_path is not None:
            utils.split_and_move_tar(tar_path, cfg.downlink_compressed_split)

    # Package compressed raws for downlinking.
    if cfg.downlink_compressed_raws and raw_compressor is not None:
        tar_path = utils.package_files_for_downlinking(cfg.raw_compression_type, cfg.downlink_log_if_no_images)

        if tar_path is not None:
            utils.split_and_move_tar(tar_path, cfg.downlink_compressed_split)


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
    
    logger = setup_logger('smartcam_logger', LOG_FILE, formatter, level=logging.INFO)

    # Start the app.
    run_experiment()