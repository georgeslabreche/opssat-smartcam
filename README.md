![OPS-SAT SmartCam Logo](https://raw.githubusercontent.com/georgeslabreche/opssat-smartcam/main/docs/ops-sat_smartcam_logo_transparentbg.png?raw=true)

# Background
The OPS-SAT SmartCam is an image acquisition and classification app for the European Space Agency's [OPS-SAT](https://www.esa.int/Enabling_Support/Operations/OPS-SAT_your_flying_laboratory) spacecraft. An acquired image can go through a pipeline of multiple image classification models that are applied in a sequence.

The app features geospatial awareness with the ability to acquire images when the spacecraft is located above pre-defined areas of interests that are described as polygons and/or multi-polygons in a GeoJSON file. 

1. [Neural Networks](https://github.com/georgeslabreche/opssat-smartcam#neural-networks)
2. [Contribute](https://github.com/georgeslabreche/opssat-smartcam#contribute)
3. [How does it work?](https://github.com/georgeslabreche/opssat-smartcam#how-does-it-work)
4. [Configuration](https://github.com/georgeslabreche/opssat-smartcam#configuration)
5. [Image Metadata](https://github.com/georgeslabreche/opssat-smartcam#image-metadata)

## 1. Neural Networks
The app can apply any .tflite neural network image classification model file trained with TensorFlow. The default model's labels are "earth", "edge", and "bad". The SmartCam's image classification program [uses the TensorFlow Lite C API for model inference](https://github.com/georgeslabreche/tensorflow-opssat-smartcam). Tensorflow Lite inference is thus available to any experimenter without being restricted to image classification. 

## 2. Contribute
Ways to contribute:
- Train a model that can be plugged into the SmartCam's image classification pipeline.
- Develop your own experiment that is unrelated to SmartCam and image classification but makes use of the SmartCam's underlying Tensorflow Lite inference program.
- OPS-SAT is your flying laboratory: come up with your own experiment that is unrelated to the SmartCam app or the AI framework that powers it.

Join the [OPS-SAT community platform](https://opssat1.esoc.esa.int/) and apply to become an experimenter, it's quick and easy! 
## 3. How does it work?
### 3.1. Overview
The SmartCam's app configuration is set in the config.ini file. The gist of the application's logic is as follow:

1. Acquires ims_rgb (raw) and png image files using the spacecraft's HD camera.
2. Creates a thumbnail jpeg image.
3. Creates an input jpeg image for the image classifier.
4. Labels the image using the entry point model file specified by *entry_point_model* in config.ini.
5. If the applied label is part of the model's *labels_keep* in config.ini then label the image further with the next model in image classification pipeline.
6. Repeat step 5 until either the applied label is not part of the current model's configured *labels_keep* or until the last model of the pipeline has been applied.
7. The labeled image is moved into the experiment and the filestore's toGround folders depending on the keep images and downlink configurations set in config.ini.
8. Repeat steps 1 through 7 until the image acquisition loop as gone through the number of iterations set by *gen_number* in config.ini.

### 3.2. Building an image classification pipeline
1. Each model consists of a .tflite and a labels.txt file located in a model folder under `/home/exp1000/models`, e.g: `/home/exp1000/models/default` and `/home/exp1000/models/cloud_detection`.
2. Create a config.ini section for each model. Prefix the section name with `model_`, e.g. `[model_default]` and `[model_cloud_detection]`.
3. Each model's config section will specify which label to keep via the *labels_keep* property. For instance, if the default model can label an image as either "earth", "edge", or "bad", but we only want to keep images classified with the first two labels, then `labels_keep = ["earth", "edge"]`.
4. If another image classification needs to follow up after an image was previously classified with a certain label, then the follow up model name can be appended following a colon. E.g. `["earth:cloud_detection", "edge"]`.
5. The entry point model that will be the first model applied in the image classification pipeline is specified in the config.ini's *entry_point_model* property, e.g. `entry_point_model = default`. 

## 4. Configuration
This section describes the app's configuration parameters in the `config.ini` file.
### 4.1. General
- *downlink_log_if_no_images* - indicate whether or not the log file(s) should be downlinked even if no images are labeled for downlink (yes/no).
- *entry_point_model* - the first image classification model to apply in the model pipeline.
- *downlink_thumbnails* - indicate whether or not thumbnails should be downlinked (yes/no).
- *downlink_compressed_raws* - indicate whether or not raws should be compressed and downlinked (yes/no).
- *downlink_compressed_split* - maximum downlink file size (bytes). File is split if this limit is exceeded. Uses the [split](https://man7.org/linux/man-pages/man1/split.1.html) command.
- *raw_compression_type* - which compression algorithm to apply on raw image files.
- *collect_metadata* - collect metadata into a CSV file (yes/no).
- *tle_path* - path to the TLE file.
- *quota_toGround* - experiment's toGround folder size limit (KB). Image acquisition is skipped if this limit is exceeded.

### 4.2. Image Acquisition
There are two types of image acquisition that can beet set: Polling or Area-of-Interest (AOI):
- Polling: acquire images in a loop that begins at the experiment start time.
- AOI: acquire images whenever the spacecraft is above an area of interest, during daytime, as defined by polygon shapes in a GeoJSON file.

#### 4.2.1. AOI GeoJSON Files
- The default AOI GeoJSON files defines multipolygon representations of all continents except Antarctica. 
- Use [geojson.io](https://geojson.io) to define custom AOI polygons for the app to consume.
- Use [mapshaper](https://mapshaper.org/) to simplify GeoJSON files onbtained from third-party providers in order to keep the file sizes small.
- Coordinates with high precision floating point numbers do not contribute much and are best avoided in favour of reduced GeoJSON file size.

#### 4.2.2. Camera Settings
- *cam_exposure* - exposure value (in milliseconds).
- *cam_gains* - rgb gains (e.g. [8, 8, 8]).

#### 4.2.3. Acquisition Type
- *gen_type* - can be either `aoi` or `poll` for "area-of-interest" or "polling", respectively.
- *gen_interval_default* - wait time between image acquisition loop iterations (in seconds).
- *gen_interval_throttle* - wait time between image acquisition loop iterations when a label of interest has been applied to the previously acquired image (in seconds).
- *gen_number* - number of image acquisitions.
- *gen_geojson* - path of the GeoJSON file with polygons defining areas of interest for image acquisition.

### 4.3. Images
- *raw_keep* - flag if the raw image file should be kept.
- *png_keep* - flag if the png image file should be kept.
- *jpeg_scaling* - scaling factor applied on the png file when generating the jpeg thumbnail.
- *jpeg_quality* - png to jpeg conversion quality level.
- *jpeg_processing* - image processing to apply when generating jpeg thumbnail (none, pnmnorm, or pnmhisteq).

### 4.4. Model
- *tflite_model* - path of the TensorFlow Lite neural network mode file.
- *labels* - path of the labels text file.
- *labels_keep* - only downlink images that are classified with these labels.
- *input_height* - scaled height of the image input that will be fed into the neural network.
- *input_width* - scaled width of the image input that will be fed into the neural network. 
- *input_mean* - mean of the image input.
- *input_std* - standard deviation of the image input.
- *confidence_threshold* - minimum confidence level required to apply the label predicted by the neural network model.

### 4.5. Raw Image Compression
#### 4.5.1. FAPEC
The FAPEC compression binary provided by [DAPCOM DataServices](dapcom.es) and not included in this repository. The compressor can only be used with a valid license (free of charge if exclusively used for OPS-SAT purposes). Free decompression licenses (with some limitations) can be obtained from the DAPCOM website or upon request to [fapec@dapcom.es](fapec@dapcom.es).

- *chunk* - chunk size.
- *threads* - number of threads.
- *dtype* - dtype.
- *band* - input band.
- *losses* - losses (0 is lossless).
- *meaningful_bits* - meaningful bits.
- *lev* - inter-band decorrelation adaptiveness.

#### 4.5.2. Others
No other image compression algorithms are currently supported.

## 5. Image Metadata
A CSV file is created and downlinked when *collect_metadata* is set to `yes`. Each row contains metadata for an image acquired during the SmartCam app's execution. Metadata for images that were discarded are also included. The following information is collected:

- *filename* - name of the image file.
- *label* - final label applied by the image classification pipeline.
- *confidence* - confidence level of the applied label.
- *keep* - whether the image was kept or not based on the applied label.
- *gain_r* - red gain value set for the camera's RGB gain setting.
- *gain_g* - green gain value set for the camera's RGB gain setting.
- *gain_b* - blue gain value set for the camera's RGB gain setting.
- *exposure* - camera exposure setting, in milliseconds.
- *acq_ts* - image acquisition timestamp, in milliseconds.
- *acq_dt* - image acquisition datetime.
- *ref_dt* - TLE reference datetime. The on-board TLE file is used to project the latitude and longitude of the spacecraft's groundtrack location, as well as its altitude, based on the image acquisition's timestamp.
- *tle_age* - age of the reference TLE.
- *lat* - latitude of the spacecraft's groundtrack location.
- *lng* - longitude of the spacecraft's groundtrack location.
- *h* - altitude of the spacecraft's orbit, i.e. height above Earth's surface.
- *tle_ref_line1* - line 1 of the reference TLE.
- *tle_ref_line2* - line 2 of the reference TLE.


![OPS-SAT Mission Patch](https://raw.githubusercontent.com/georgeslabreche/opssat-smartcam/main/docs/ops-sat_mission_patch.png?raw=true)
