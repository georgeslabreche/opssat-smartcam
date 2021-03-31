## Background

- Use Transfer Learning to train models that classify OPS-SAT thumbnail images.
- Images acquired by spacecraft's on-board camera are hosted in the [OPS-SAT Community Platform](https://opssat1.esoc.esa.int/).
- The trained models can be used with the [SmartCam](https://github.com/georgeslabreche/opssat-smartcam) app.
- Further details and examples on Transfer Learning with TensorFlow can be found [here](https://github.com/tensorflow/hub/tree/master/tensorflow_hub/tools/make_image_classifier) and [here](https://github.com/tensorflow/hub/blob/master/examples/colab/tf2_image_retraining.ipynb).


1. [Installation](https://github.com/georgeslabreche/opssat-smartcam/train#1-installation)
2. [Training a Model](https://github.com/georgeslabreche/opssat-smartcam/train#2-training-a-model)
3. [Known Issues](https://github.com/georgeslabreche/opssat-smartcam/train#3-known-issues)

## 1. Installation

1. Create the virtual environment: `python3 -m venv venv`
2. Source into the virtual environment: `source venv/bin/activate`
3. Update pip3: `pip3 install -U pip`
4. Update setuptools: `pip3 install -U setuptools`
5. Install tensorflow requirements: `pip3 install -r requirements.txt`

## 2. Training a Model
The model is trained with the `make_image_classifier` command. Usage instructions can be found [here](https://github.com/tensorflow/hub/tree/master/tensorflow_hub/tools/make_image_classifier)along with descriptions of the available parameters and hyperparameters. All example commands in this section assume a model named `my_model_name`.

### 2.1. Directories

1. Create the directories used to train and validate the model: `./create_dirs.sh my_model_name`
2. Put all pre-labeled images in the `repo/my_model_name/data/all` directory. 
3. Split all images in two groups: 75% training data and 25% test data: `python3 split_data.py my_model_name 25`
4. Check that the data has been split correctly by peaking into `repo/my_model_name/data/training` and `repo/my_model_name/data/test`.

### 2.2. Training

Delete the summaries data that was created during a previous training:
```
rm -rf repo/my_model_name/summaries
```

Run the `make_image_classifier` command on the training data set:

```bash
make_image_classifier \
  --image_dir repo/my_model_name/data/training \
  --tfhub_module https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4 \
  --saved_model_dir repo/my_model_name \
  --labels_output_file repo/my_model_name/labels.txt \
  --tflite_output_file repo/my_model_name/tflite_model.tflite \
  --train_epochs 5 \
  --summaries_dir repo/my_model_name/summaries
```

Monitor the model training on TensorBoard at `http://localhost:6006/`:
```
tensorboard --logdir repo/my_model_name/summaries
```

The scalars in TensorBoard for epoch accuracy and epoch loss are updated after each processed epoch.

### 2.4. Testing
Use the trained model' to classify all images in `repo/my_model_name/data/test` by running the following Python script:

```bash
python3 batch_label_images.py my_model_name
```

What the script does:
1. Invokes the `label_image.py` script to classify each image. This script uses the trained model to predict a label for a given image.
2. The classified images are copied to `repo/my_model_name/data/classification`.
3. A classification confidence log is saved as a CSV file in `repo/my_model_name/data/classification/confidences.csv`.

Analyze the classification accuracies for each label to test the model with respect to your performance criteria:

```python
python3 calc_perf.py my_model_name
```

If classifications are not accurate enough then try training again using a higher epoch value. **Note that training with more epochs does not necessarily produce models with higher inference accuracies against your test data.** This is due to [overfitting](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit): _"If you train for too long though, the model will start to overfit and learn patterns from the training data that don't generalize to the test data. We need to strike a balance."_

### 2.4. Tools

TensorFlow provides a bunch of [useful tools](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools) to evaluate the trained .tflite model: benchmark, inspect, visualize, and more. [Model optimization](https://www.tensorflow.org/lite/performance/model_optimization) is also possible.
## 3. Known Issues

The following error may be thrown while running `make_image_classifier`:

```bash
 File "/home/georges/dev/tf/train/make_image_classifier/venv/lib/python3.6/site-packages/PIL/ImageFile.py", line 260, in load 
 "image file is truncated" 
 OSError: image file is truncated (49 bytes not processed) 
```

This can be resolved by editing the `ImageFile.py` file, search for `LOAD_TRUNCATED_IMAGES` and edit it so that it is set to `True`:

```python
LOAD_TRUNCATED_IMAGES = True
```

Solution taken from [here](https://stackoverflow.com/a/23575424/4030804).