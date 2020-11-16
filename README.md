# OPS-SAT SmartCam
An image acquisition and classification app for the European Space Agency's [OPS-SAT](https://www.esa.int/Enabling_Support/Operations/OPS-SAT_your_flying_laboratory) spacecraft.

## Background
The SmartCam app enables autonomous on-board decision making in downlink prioritization. It uses artificial intelligence (AI) to identify images of interest as soon as they are captured by the spacecraft's camera. During its commissioning phase, OPS-SAT downlinked over 4,500 thumbnail images. These pictures were used as training data for [TensorFlow Lite](https://www.tensorflow.org/lite) to develop an image classification convolutional neural network (CNN) model. 

The TensorFlow Lite inference API is meant for machine learning (ML) models on mobile and IoT devices but OPS-SAT's powerful on-board computer makes it possible to run it in space. Since image classification is a common problem, the model trained for the SmartCam app benefitted from [transfer Learning (TL)](https://www.tensorflow.org/tutorials/images/transfer_learning) to greatly reduce the complexity, time, and processing power required to train the model.

The app was developed with accessibility and re-useability in mind; anyone can uplink their own TensorFlow Lite image classification model for the SmartCam to use without having to develop their own app. A future version will allow multiple models to be chained into an image classification pipeline, with branching options, opening exciting possibilities for crowdsourced space segment AI operations.

## Inference
The SmartCam app is written in Python. The inference program it invokes is written in C and can be found [here](https://github.com/georgeslabreche/tensorflow-opssat-smartcam).

## Neural Networks
The app can apply any .tflite neural network image classification model file trained with TensorFlow. The default model labels the images acquired by the spacecraft's camera as either "earth", "edge", or "bad". 

## How does it work?
The desired app configurations are set in the config.ini file. Starting the app triggers the following sequence of operations:

1. Acquires ims_rgb (raw) and png image files using the spacecraft's HD camera.
2. Creates a thumbnail jpeg image.
3. Creates a neural network input jpeg image.
4. Labels the input image using the neural network model file specified in the config.ini file.
5. Repeat steps 2 through 5 for as many times specified in the config.ini file.
6. Move the labeled thumbnails and log file into the app's toGround folder for downlinking.

## Configuration
Consult the app's config.ini file for the default configuration values,

### General
- *downlink_log_if_no_images* - flag if the log file(s) should be downlinked even if no images are labeled for downlink.

### Model
- *tflite_model* - path of the TensorFlow Lite neural network mode file.
- *labels* - path of the labels text file.
- *labels_keep* - only downlink images that are classified with these labels.
- *input_height* - scaled height of the image input that will be fed into the neural network.
- *input_width* - scaled width of the image input that will be fed into the neural network. 
- *input_mean* - mean of the image input.
- *input_std* - standard deviation of the image input.
- *confidence_threshold* - minimum confidence level required to apply the label predicted by the neural network model.

### Image Acquisition
- *gen_interval* - image acquisitions period in seconds.
- *gen_number* - number of image acquisition interations.
- *gen_exposure* - camera exposure value.
- *gen_gains* - rgb gains.

### Images
- *raw_keep* - flag if the raw image file should be kept.
- *png_keep* - flag if the png image file should be kept.
- *jpeg_scaling* - scaling factor applied on the png file when generating the jpeg thumbnail.
- *jpeg_quality* - png to jpeg conversion quality level.
- *jpeg_processing* - image processing to apply when generating jpeg thumbnail (none, pnmnorm, or pnmhisteq).
