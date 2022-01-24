![OPS-SAT SmartCam Logo](https://raw.githubusercontent.com/georgeslabreche/opssat-smartcam/main/docs/ops-sat_smartcam_logo_transparentbg.png?raw=true)

# Background
The SmartCam software on the [OPS-SAT](https://www.esa.int/Enabling_Support/Operations/OPS-SAT_your_flying_laboratory) spacecraft is the first use of Artificial Intelligence (AI) by the European Space Agency (ESA) for autonomous planning and scheduling on-board a flying mission. The software's geospatial capability autonomously triggers image acquisitions when the spacecraft is above areas of interest. Inferences from on-board Machine Learning (ML) models classify the captured pictures for downlink prioritization. This is made possible by the spacecraft's powerful processors, capable of running open-source software originally developed for terrestrial systems. Notably, with the [GEOS Geometry Engine](https://libgeos.org/) for geospatial computations and the [TensorFlow Lite](https://www.tensorflow.org/lite) framework for ML model inferences. Additional image classification can be enabled with unsupervised learning using [k-means clustering](https://github.com/georgeslabreche/kmeans-image-clustering/tree/opssat). These features provide new perspectives on how space operations can be designed for future missions given greater in-orbit compute capabilities.

The SmartCam's image classification pipeline is made "openable" by allowing it to be constructed from crowdsourced trained ML models. These third-party models can be uplinked to the spacecraft and chained into a sequence with configurable branching rules for hyper-specialized classification and subclassification through an autonomous decision-making tree. This mechanism enables open innovation methods to extend on-board ML beyond its original mission requirement while stimulating knowledge transfer from established AI communities into space applications. The use of an industry standard ML framework de-risks and accelerate developing AI for future missions by broadening OPS-SAT's accessibility to AI experimenters established outside of the space sector.

# Citation
We appreciate citations if you reference this work in our upcoming scientific publication. Thank you!

## APA
Labrèche, G., Evans, D., Marszk, D., Mladenov, T., Shiradhonkar, V., Soto, T., & Zelenevskiy, V. (2022). OPS-SAT Spacecraft Autonomy with TensorFlow Lite, Unsupervised Learning, and Online Machine Learning. _2022 IEEE Aerospace Conference._

## BibTex
```bibtex
@article{LabrecheIEEEAeroconf2022,
  title={OPS-SAT Spacecraft Autonomy with TensorFlow Lite, Unsupervised Learning, and Online Machine Learning},
  author={Georges Labrèche and David Evans and Dominik Marszk and Tom Mladenov and Vasundhara Shiradhonkar and Tanguy Soto and Vladimir Zelenevskiy},
  journal={2022 IEEE Aerospace Conference},
  year={2022}
}
```

# Instructions
**Table of Contents:**
1. [Neural Networks](https://github.com/georgeslabreche/opssat-smartcam#1-neural-networks)
2. [Contribute](https://github.com/georgeslabreche/opssat-smartcam#2-contribute)
3. [How It Works](https://github.com/georgeslabreche/opssat-smartcam#3-how-it-works)
4. [Configuration](https://github.com/georgeslabreche/opssat-smartcam#4-configuration)
5. [Image Metadata](https://github.com/georgeslabreche/opssat-smartcam#5-image-metadata)

## 1. Neural Networks
The app can use any .tflite neural network image classification model file trained with TensorFlow. 
## 1.1. Inference
The default model's labels are "earth", "edge", and "bad". The SmartCam's image classification program [uses the TensorFlow Lite C API for model inference](https://github.com/georgeslabreche/tensorflow-opssat-smartcam). Tensorflow Lite inference is thus available to any experimenter without being restricted to image classification.

## 1.2. Training New Models
Scripts and instructions to train new models are available [here](https://github.com/georgeslabreche/opssat-smartcam/tree/main/train).

## 2. Contribute
Ways to contribute:
- Train a model that can be plugged into the SmartCam's image classification pipeline.
- Develop your own experiment that is unrelated to SmartCam and image classification but makes use of the SmartCam's underlying Tensorflow Lite inference program.
- OPS-SAT is your flying laboratory: come up with your own experiment that is unrelated to the SmartCam app or the AI framework that powers it.

Join the [OPS-SAT community platform](https://opssat1.esoc.esa.int/) and apply to become an experimenter, it's quick and easy! 

## 3. How It Works
The app is designed to run on the Satellite Experimental Processing Platform (SEPP) payload onboard the OPS-SAT spacecraft. The SEPP is a powerful ALTERA Cyclone V with a 800 MHz CPU clock and 1GB DDR3 RAM. 

### 3.1. Overview
The SmartCam's app configuration is set in the config.ini file. The gist of the application's logic is as follow:

1. Acquires ims_rgb (raw) and png image files using the spacecraft's HD camera.
2. Creates a thumbnail jpeg image.
3. Creates an input jpeg image for the image classifier.
4. Labels the image using the entry point model file specified by *entry_point_model* in config.ini.
5. If the applied label is part of the model's *labels_keep* in config.ini then label the image further with the next model in image classification pipeline.
6. Repeat step 5 until either the applied label is not part of the current model's configured *labels_keep* or until the last model of the pipeline has been applied.
7. The labeled image is moved into the experiment and the filestore's toGround folders depending on the keep images and downlink configurations set in config.ini.
8. Subclassify the labeled images into cluster folders via k-means clustering (or train the clustering model if not enough training images have been collected yet).
9. Repeat steps 1 through 8 until the image acquisition loop as gone through the number of iterations set by *gen_number* in config.ini.

For certain operations the app invokes external executable binaries that are packaged with the app. The are included in [this bin folder](https://github.com/georgeslabreche/opssat-smartcam/tree/issue_13_mock_image_acquisition/home/exp1000/bin). Their source codes are hosted in separate repositories:
- [Image resizing](https://github.com/georgeslabreche/image-resizer).
- [TensorFlow inference](https://github.com/georgeslabreche/tensorflow-opssat-smartcam).
- [K-means image clustering and image segmentation (feature extraction)](https://github.com/georgeslabreche/kmeans-image-clustering).

### 3.2. Installation
The app can run on a local development environment (64-bit) as well as onboard the spacecraft's SEPP processor (ARM 32-bit). For the former, the app reads its configuration parameters from the *config.dev.ini" file whereas for the latter it reads them from the *config.ini* file. 

#### 3.2.1. Local Development Environment
These instruction are written for Ubuntu and were tested on Ubuntu 18.04 LTS. Install development tools:
```bash
sudo apt install python3-dev
sudo apt install virtualenv
```

Create the symbolic links for the TensorFlow Lite shared objects. Execute the following bash script from the project's home directory:
```bash
./scripts/create_local_dev_symlinks.sh
```

Install a Python virtual environment and the Python package dependencies. From the project's home directory:
```bash
cd home/exp1000/
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

Edit the *smartcam.py* file to enable debug mode and indicate that the app must execute binaries compiled for the 64-bit local dev environment. These binaries were built with the k8 architecture. Enabling debug mode simply generates mock data (e.g. acquired pictures) in the absence of spacecraft hardware (e.g. the onboard camera).
```python
DEBUG = True
DEBUG_ARCH = 'k8'
```

Before running the app make sure that the virutal environment is still active. If it isn't then re-execute `source venv/bin/activate`. Run the app:
```bash
python3 smartcam.py
```

#### 3.2.2. Onboard the Spacecraft
The SmarCam app and its dependencies are packaged for deployment as opkg ipk files, ready to be installed in the SEPP via the `opkg install` command.

##### 3.2.2.1. Dependencies
The SEPP runs the Ångström distribution of Linux. The following packages are dependencies that need to be installed in SEPP's Linux operating system prior to installing and running the app. They can be found in the `deps` directory of this repository:
- **ephem 3.7.6.0:** Python package for high-precision astronomy computations. 
- **Shapely 1.7.0:** Python package for manipulation and analysis of planar geometric objects. 
- **libgeos 3.5.1:** Geometry engine for Geographic Information Systems - C++ Library.
- **libgeos-c 1.9.1:** Geometry engine for Geographic Information Systems - C Library.

Other dependencies are the *tar* and *split* programs that are invoked by the App.

##### 3.2.2.2. The App
The SmartCam app has also been packaged for installation via opkg. The ipk files for tagged releases are available in the Tags section of this repository.


### 3.3. Building an Image Classification Pipeline
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
- *max_error_count* - maximum number of errors thrown before exiting the image acquisition loop.

### 4.2. Image Acquisition
There are two types of image acquisition that can beet set: Polling or Area-of-Interest (AOI):
- Polling: acquire images in a loop that begins at the experiment start time.
- AOI: acquire images whenever the spacecraft is above an area of interest, during daytime, as defined by polygon shapes in a GeoJSON file.

#### 4.2.1. Area-of-Interest GeoJSON Files
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

### 4.4. Model
- *tflite_model* - path of the TensorFlow Lite neural network mode file.
- *labels* - path of the labels text file.
- *labels_keep* - only downlink images that are classified with these labels.
- *input_height* - scaled height of the image input that will be fed into the neural network.
- *input_width* - scaled width of the image input that will be fed into the neural network. 
- *input_mean* - mean of the image input.
- *input_std* - standard deviation of the image input.
- *confidence_threshold* - minimum confidence level required to apply the label predicted by the neural network model.

### 4.5. Clustering
- *cluster* - flag to enable or disable image clustering with k-means.
- *cluster_for_labels* - labels of images classified by TensorFlow Lite that should be subclassified with k-means clustering.
- *cluster_k* - number of clusters.
- *cluster_collect_threshold* - number of images to collect as training data before training the clustering model.
- *cluster_img_types* - list of image files type to move into the cluster folders during clustering.

#### 4.4.1. Data Normalization
A note on what *input_mean* and *input_std* are for, taken verbatim from [this blogpost](https://medium.com/@joel_34096/k-means-clustering-for-image-classification-a648f28bdc47):

> Since the dataset contains a range of values from 0 to 255, the dataset has to be normalized. Data Normalization is an important preprocessing step which ensures that each input parameter (pixel, in this case) has a similar data distribution. This fastens the process of covergence while training the model. Also Normalization makes sure no one particular parameter influences the output significantly. Data normalization is done by subtracting the mean from each pixel and then dividing the result by the standard deviation. The distribution of such data would resemble a Gaussian curve centered at zero. For image inputs we need the pixel numbers to be positive. So the image input is divided by 255 so that input values are in range of [0,1].

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
