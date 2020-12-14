# OPS-SAT SmartCam
An image acquisition and classification app for the European Space Agency's [OPS-SAT](https://www.esa.int/Enabling_Support/Operations/OPS-SAT_your_flying_laboratory) spacecraft. An acquired image can go through a pipeline of multiple image classification models that are applied in a sequence.

## Neural Networks
The app can apply any .tflite neural network image classification model file trained with TensorFlow. The default model labels the images acquired by the spacecraft's camera as either "earth", "edge", or "bad". 

## Contribute
The SmartCam's image classification program [uses the TensorFlow Lite C API for model inference](https://github.com/georgeslabreche/tensorflow-opssat-smartcam). Tensorflow Lite inference is thus available to any experimenter without being restricted to image classification. Ways to contribute:
- Train an model that can be plugged into the SmartCam's image classification pipeline.
- Develop your own experiment that is unrelated to SmartCam and image classification but makes use of the SmartCam's underlying Tensorflow Lite inference program.
- OPS-SAT is your flying laboratory: come up with your own experiment that is unrelated to the SmartCam app or the AI framework that powers it.

Join the [OPS-SAT community platform](https://opssat1.esoc.esa.int/) and apply to become an experimenter, it's quick and easy! 
## How does it work?
### Overview
The desired app configurations are set in the config.ini file. The gist of the application's logic is as follow:

1. Acquires ims_rgb (raw) and png image files using the spacecraft's HD camera.
2. Creates a thumbnail jpeg image.
3. Creates a neural network input jpeg image.
4. Labels the image using the entry point neural network model file specified by *entry_point_model* in config.ini.
5. If the applied label is part of the model's *labels_keep* in config.ini then label the image further with the next model in model pipeline.
6. Repeat step 5 until either the applied label is not part of the current model's configured *labels_keep* or until the last model of the pipeline has been applied.
7. The labeled images are moved into the experiment and the filestore's toGround folders depending on the keep images and downlink configurations set in config.ini.

### Building an image classification pipeline
1. Each model consists of a .tflite and a labels.txt file located in a model folder under `/home/exp1000/models`, e.g: `/home/exp1000/models/default` and `/home/exp1000/models/cloud_detection`.
2. Create a config.ini section for each model. Prefix the section name with `model_`, e.g. `[model_default]` and `[model_cloud_detection]`.
3. Each model's config section will specify which label to keep via the *labels_keep* property. For instance, if the default model can label an image as either "earth", "edge", or "bad", but we only want to keep images classified with the first two labels, then `labels_keep = ["earth", "edge"]`.
4. If another image classification needs to follow up after an image was previously classified with a certain label, then the follow up model name can be appended following a colon. E.g. `["earth:cloud_detection", "edge"]`.
5. The entry point model that will be the first image classification applid in the model pipeline is specified in the config.ini's *entry_point_model* property, e.g. `entry_point_model = default`. 

## Configuration
Consult the app's config.ini file for the default configuration values.

### General
- *downlink_log_if_no_images* - indicate whether or not the log file(s) should be downlinked even if no images are labeled for downlink (yes/no).
- *entry_point_model* - the first image classification model to apply in the model pipeline.
- *downlink_thumbnails* - indicate whether or not thumbnails should be downlinked (yes/no).
- *downlink_compressed_raws* - indicate whether or not raws should be compressed and downlinked (yes/no).
- *downlink_compressed_split* - maximum downlink file size (bytes). File is split if this limit is exceeded. Uses the [split](https://man7.org/linux/man-pages/man1/split.1.html) command.
- *raw_compression_type* - which compression algorithm to apply on raw image files.
- *collect_metadata* - collect metadata into a CSV file (yes/no).
- *tle_path* - path to the TLE file.
- *quota_toGround* - experiment's toGround folder size limit (KB). Image acquisition is skipped if this limit is exceeded.

### Image Acquisition
- *gen_interval* - default image acquisition period (in seconds).
- *gen_interval_throttle* - image acquisition period used when a label of interest has been applied to the previously acquired image (in seconds).
- *gen_number* - number of image acquisition interations.
- *gen_exposure* - camera exposure value.
- *gen_gains* - rgb gains.

### Images
- *raw_keep* - flag if the raw image file should be kept.
- *png_keep* - flag if the png image file should be kept.
- *jpeg_scaling* - scaling factor applied on the png file when generating the jpeg thumbnail.
- *jpeg_quality* - png to jpeg conversion quality level.
- *jpeg_processing* - image processing to apply when generating jpeg thumbnail (none, pnmnorm, or pnmhisteq).

### Model
- *tflite_model* - path of the TensorFlow Lite neural network mode file.
- *labels* - path of the labels text file.
- *labels_keep* - only downlink images that are classified with these labels.
- *input_height* - scaled height of the image input that will be fed into the neural network.
- *input_width* - scaled width of the image input that will be fed into the neural network. 
- *input_mean* - mean of the image input.
- *input_std* - standard deviation of the image input.
- *confidence_threshold* - minimum confidence level required to apply the label predicted by the neural network model.

### Raw Image Compression
#### Fapec
- *chunk* - chunk size.
- *threads* - number of threads.
- *dtype* - dtype.
- *band* - input band.
- *losses* - losses (0 is lossless).
- *meaningful_bits* - meaningful bits.
- *lev* - inter-band decorrelation adaptiveness.

#### Others
No other image compression algorithms are currently supported.