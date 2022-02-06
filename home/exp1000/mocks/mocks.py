import time
import csv
from pathlib import Path
import glob
import os
import shutil

class MockImageMetaData:

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


    def __init__(self, base_path, tle_path, gains, exposure):
        self.base_path = base_path
        self.gains = gains
        self.exposure = exposure


    def get_groundtrack_coordinates(self):
        """Get coordinates of the geographic point beneath the satellite."""
        return {
            'lat': 40.7306 * 3.14/180,
            'lng': -73.9352 * 3.14/180,
            'dt': round(time.time() * 1000)
        } 


    def is_daytime(self, ephem_lat, ephem_lng, dt):
        """Check if it's daytime at the given location for the given time."""
        return True


    def collect_metadata(self, filename_png, label, confidence, keep):
        """Collect metadata for the image acquired at the given timestamp."""

        # The filename without the path and without the extension
        filename = filename_png.replace(self.base_path + "/", "").replace(".png", "")

        # Return the mock metadata.
        return {
            'filename': filename,                 # Filename without extension.
            'label': label,                       # Label applied to the image by the image classifier.
            'confidence': confidence,             # Confidence level of th the applied label.
            'keep': keep,                         # Whether the image was kept or not.
            'gain_r': self.gains[0],              # Camera gain setting for red channel.
            'gain_g': self.gains[1],              # Camera gain setting for green channel.
            'gain_b': self.gains[2],              # Camera gain setting for blue channel.
            'exposure': self.exposure,            # Camera exposure setting (ms).
            'acq_ts': 'mock',                     # Image acquisition timestamp.
            'acq_dt': 'mock',                     # Image acquisition datetime.
            'ref_dt': 'mock',                     # Reference epoch.
            'tle_age': 'mock',                    # TLE age (days).
            'lat': 'mock',                        # Latitude (deg).
            'lng': 'mock',                        # Longitude (deg).
            'h': 'mock',                          # Geocentric height above sea level (m).
            'tle_ref_line1': 'mock',              # Reference TLE line 1.
            'tle_ref_line2': 'mock'               # Rererence TLE line 2.
        }


    def write_metadata(self, csv_filename, metadata):
        """Write collected metadata into a CSV file."""

        # If file exists it means that it has a header already.
        existing_csv_file = Path(csv_filename)
        has_header = existing_csv_file.is_file()

        # Open CSV file and write an image metadata row for the acquired image.
        with open(csv_filename, 'a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.FIELD_NAMES)

            # Write header if it's not already there:
            if has_header is False:
                writer.writeheader()

            # Write image metadata row.
            writer.writerow(metadata)


class MockHDCamera:

    def __init__(self):
        # Get picture file urls.
        self.pics = glob.glob("mocks/pictures/*.png")

        # Get number of pictures.
        self.pics_size = len(self.pics)

        # Start index.
        self.pics_index = 0

    def acquire_image(self):
        """Acquire an image with the on-board camera."""

        # Get picture file url at current file index of the mock picture folder.
        pic_url = self.pics[self.pics_index]

        # Increment file index for next mock picture acquisition.
        self.pics_index = self.pics_index + 1

        # Reset counter to zero.
        if self.pics_index >= self.pics_size:
            self.pics_index = 0

        # Get the picture's file name with the directory path.
        pic_filename = os.path.basename(pic_url)

        # Create a mock RAW ims_rgb image file by change the .png extention to .ims_rgb.
        # Copy the RAW image to the home directory.
        # TODO: Use real RAW image files?
        shutil.copyfile(pic_url, pic_filename.replace('.png', '.ims_rgb'))

        # Copy the PNG image to the home directory.
        shutil.copyfile(pic_url, pic_filename)

        # Return pic file name.
        return pic_filename