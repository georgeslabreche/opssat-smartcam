## Background

- Use Transfer Learning to train models that classify OPS-SAT thumbnail images.
- Images acquired by spacecraft's on-board camera are hosted in the [OPS-SAT Community Platform](https://opssat1.esoc.esa.int/).
- The trained models can be used with the [SmartCam](https://github.com/georgeslabreche/opssat-smartcam) app.
- Further details and examples on Transfer Learning with TensorFlow can be found [here](https://github.com/tensorflow/hub/tree/master/tensorflow_hub/tools/make_image_classifier) and [here](https://github.com/tensorflow/hub/blob/master/examples/colab/tf2_image_retraining.ipynb).


1. [Installation](https://github.com/georgeslabreche/opssat-smartcam/tree/main/train#1-installation)
2. [Training a Model](https://github.com/georgeslabreche/opssat-smartcam/tree/main/train#2-training-a-model)
3. [Known Issues](https://github.com/georgeslabreche/opssat-smartcam/tree/main/train#3-known-issues)

## 1. Installation

1. Create the virtual environment: `python3 -m venv venv`
2. Source into the virtual environment: `source venv/bin/activate`
3. Update pip3: `pip3 install -U pip`
4. Update setuptools: `pip3 install -U setuptools`
5. Install tensorflow requirements: `pip3 install -r requirements.txt`

## 2. Training a Model
The model is trained with the `make_image_classifier` command. Usage instructions can be found [here](https://github.com/tensorflow/hub/tree/master/tensorflow_hub/tools/make_image_classifier) along with descriptions of the available parameters and hyperparameters. All example commands in this section assume a model named `my_model_name`.

### 2.1. Directories

1. Create the directories used to train and test the model: `./create_dirs.sh my_model_name`
2. Put all pre-labeled images in the `repo/my_model_name/data/all` directory. 
3. Split all images in two groups: 75% training data and 25% test data: `python3 split_data.py my_model_name 25`
4. Check that the data has been split correctly by peaking into `repo/my_model_name/data/training` and `repo/my_model_name/data/test`.

Behind the scene, `make_image_classifier` will [split the training data into training and validation pieces](https://github.com/tensorflow/hub/blob/44e2e19387ed756bc7f1c6e128044f4e26a937db/tensorflow_hub/tools/make_image_classifier/make_image_classifier.py#L59). This is why we only prepare training and testing datasets and do not worry about creating a validation dataset.

Helpful videos to understand the difference between training set vs test set vs validation set:
- [Intuition: Training Set vs. Test Set vs. Validation Set](https://www.youtube.com/watch?v=swCf51Z8QDo)
- [Train, Test, & Validation Sets explained](https://www.youtube.com/watch?v=Zi-0rlM4RDs)

### 2.2. Training

Run the `make_image_classifier` command on the training dataset, specify the number of epochs as the second argument:

```bash
./train_model.sh my_model_name 100
```

Monitor the epoch training and validation loss curves on TensorBoard at `http://localhost:6006/`:
```
tensorboard --logdir repo/my_model_name/summaries
```

**Observe the validation loss curves to determine when to stop the training.** The epoch loss curves also helps determine whether or not underfitting or overfitting is occuring. [Understanding the training and validation loss curves](https://www.youtube.com/watch?v=p3CcfIjycBA) is very important to guide you into creating a robust model. 

### 2.3. Testing
Use the trained model' to classify all images in `repo/my_model_name/data/test` by running the following Python script:

```bash
python3 batch_label_images.py my_model_name
```

What the script does:
1. Invokes the `label_image.py` script to classify each image. This script uses the trained model to classify an image with a label.
2. By default, a classification is accepted if the confidence level is greater than or equal to 70%. This is configurable.
3. The classified images are copied to `repo/my_model_name/data/classification`.
4. A classification confidence log is saved as a CSV file in `repo/my_model_name/data/classification/confidences.csv`.

Analyze the classification accuracies for each label to test the model with respect to your performance criteria:

```bash
python3 calc_perf.py my_model_name
```

If classifications are not accurate enough then try training again using a higher epoch value. **Training with more epochs does not necessarily produce models with higher inference accuracies against your test data.** The accuracy of a model peaks after training for a number of epochs, and then stagnates or starts decreasing. This is due to [overfitting](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit):


 _"If you train for too long though, the model will start to overfit and learn patterns from the training data that don't generalize to the test data. We need to strike a balance [...] To prevent overfitting, the best solution is to use more complete training data. The dataset should cover the full range of inputs that the model is expected to handle. Additional data may only be useful if it covers new and interesting cases."_

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