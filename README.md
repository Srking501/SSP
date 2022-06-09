[![Newcastle Univeristy Logo](assets/images/newcastle-uni-logo.png "Newcastle Univeristy Logo")](https://www.ncl.ac.uk/)

# **SSP**

Software Stack Program (SSP). The process to automate the assessment of Machine Learning (ML) Models.

**NOTE: This is a dissertation work for Newcastle University Computer Science course in the theme of Data Science and AI.**


# Preview


## Data Processing Stack

![Data Processing Stack](assets/videos/Data_Preprocessing.gif "Data Process in Progress")


## Data Manipulation and Training of ML Models Stack

![Data Manipulation and Training of ML Models](assets/videos/Data_Man_and_Train_ML.gif "Data Manipulated and Training of ML Model is Running")


## Analytics For ML Models Stack

![Analytics For ML Models](assets/videos/Analytics_For_ML_Models.gif "The Produced Analytics")


# Installation

This final notebook was solely designed and tested on [Google Colab](https://colab.research.google.com/).

It would not likely work in your machine unless if you could modify the Google libraries imports, like [Google Drive function](https://colab.research.google.com/notebooks/io.ipynb), and use a [POSIX](https://en.wikipedia.org/wiki/POSIX) complaint system.

# Usage

See [Flowchart of the Software Stack Program](assets/images/flowchart-ssp.png) for the main logic flow of the SSP.

This software has a various number of constant variables, which each can be modified to manipulate parameters for the program, these are:

## **Models to Run**
```Python
#################
# MODELS TO RUN #
#################
YOLO_MODEL = True
RES_NET_MODEL = True
MOBILENET_MODEL = True
```
These are conditions that decides what model you want to run in the notebook - `True` means run, `False` means **do not** run.

| Variables Meanings |
|--------------------|
* `YOLO_MODEL` - to run the Tiny-YOLOv3 model.

* `RES_NET_MODEL` - to run the ResNet50V2 model.

* `MOBILENET_MODEL` - to run the MobileNetV2 model.


## **Setup and Hyper-parameters for the program to work**


```Python
##########
# SET-UP #
##########
SKIP_DATA_PROCESSING = False
CLASSES = ["Person", "No Person"]
PATH_DATA = DRIVE_PATH + "quick_preview/"
TEST_DATA = PATH_DATA + "test/"
FINAL_MODELS_PATH = PATH_DATA + "models/"
BACKUP_TENSOR = PATH_DATA + "backup/tensorflow/"
NUMPY_SAVE = PATH_DATA + "numpy/"
```
The following is the setup for the essential directories for the rest of the notebook to work.

First, it contain an option to skip the [Data Processing Stack](#data-processing-stack) if the data is already processed by switching `SKIP_DATA_PROCESSING` into `True` - the `SKIP_DATA_PROCESSING` is automatically switched into `True` when the data in `YOLO_FRAMES` (look below to the YOLO SET-UP code snippet) and `NUMPY_SAVE` has content, which is then an assumption is made that the data is already processed.

Second, `CLASSES` is a list/array  of the ML class names labels.

Last, `PATH_DATA`, `TEST_DATA`, `FINAL_MODELS_PATH`, `BACKUP_TENSOR` and `NUMPY_SAVE` are paths for the directories that makes the notebook to setup the data files for training.

| Variables Meanings |
|--------------------|
* `SKIP_DATA_PROCESSING` - look above for important details. Skips the [Data Processing Stack](#data-processing-stack) when boolean value is `True`.

* `CLASSES` - a list/array of string class names labels.

* `PATH_DATA` - the directory where the data is present. The rest of the directories are designed on top of the `PATH_DATA`, where they are all in inside of the `PATH_DATA` directory.

* `TEST_DATA` - directory for any extra video or image data to test the ML model performance (has not been used in the notebook, thus manually input is needed to use it).

* `FINAL_MODELS_PATH` - the final trained models on the inputted dataset.

* `BACKUP_TENSOR` - the checkpoints weights of the training of [Tensorflow](https://www.tensorflow.org/) and [Keras](https://keras.io/api/applications/) supported models.

* `NUMPY_SAVE` - the save numpy arrays of the processed data.

```Python
NUM_BUF = 30
MIN_CONTOUR = 100
MAX_CONTOUR = 10000
# Use website to visualise how the HSV range colour selected looks  like:
# https://wamingo.net/rgbbgr/ (WARNING: it uses (360Degree, 100%, 100%) data)
# OpenCV uses (0-179, 0-255, 0-255)
#
# Trick to convert it:
# (half the degree, 255 x 1.0, 255 x 1.0)
# (Trick got from https://stackoverflow.com/a/10951189)
COLOUR_HSV_RANGE = [  # [lower bound, upper bound]
    [np.array([156, 148, 150]), np.array([179, 255, 255])],  # Red Range
    [np.array([110, 125, 125]), np.array([150, 255, 255])]  # Blue Range
]
```
These constant variables are the essential tweaks parameters to adjust the [Data Processing Stack](#data-processing-stack).

| Variables Meanings |
|--------------------|
* `NUM_BUF` - integer value for the the size of the buffer. This buffer saves the loaded frames of a video dataset where it prevents the data from being processed until it reaches the buffer size total value, for example `30` in this code snippet (this is used to reduce the amount of computation needed in [Data Processing Stack](#data-processing-stack) and minimise duplication of dataset).

* `MIN_CONTOUR` and `MAX_CONTOUR` - [contours](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html) are a curve joining all the continuous points (along the boundary), that has the same colour or intensity. Basically, when you decrease the `MIN_CONTOUR` value, you will in turn allow more smaller objects to be detected by the [Data Processing Stack](#data-processing-stack). On the other hand, when you increase the `MAX_CONTOUR` value, you will allow more bigger objects to be detected.

* `COLOUR_HSV_RANGE` - 2D array containing all the lower and upper bounds of a colour range to be detect in the [Data Processing Stack](#data-processing-stack). As stated in the variable comment, you could use the help of [wamingo website](https://wamingo.net/rgbbgr/) with the addition of [stackoverflow trick](https://stackoverflow.com/a/10951189) to figure out the [HSV](https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html) ranges needed. There is another great options is to copy the code in this [stackoverflow comment](https://stackoverflow.com/a/59906154) to have GUI window to help visualise and find the right [HSV](https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html) values for [masking](https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html).


```Python
####################
# Hyper-parameters #
####################
IMG_SIZE = 416
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
BATCH_SIZE = 20
SHUFFLE_BUFFER_SIZE = 100
EPOCHS = 10
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
```
These are hyper-parameters that are used to tweak the training process of [Tensorflow](https://www.tensorflow.org/) and [Keras](https://keras.io/api/applications/) supported ML models (see [Data Manipulation and Training of ML Models Stack](#data-manipulation-and-training-of-ml-models-stack)).

| Variables Meanings |
|--------------------|
* `IMG_SIZE` - the set horizontal and vertical pixels value for the dataset image.

* `INPUT_SHAPE` - the input shape of the dataset image for the ML models to understand and interpret in [Tensorflow](https://www.tensorflow.org/) and [Keras](https://keras.io/api/applications/).

* `BATCH_SIZE` - the collection of subset of dataset images to be inputted in a train run. So, in this a batch of `20` images are pushed in the model to be trained until it reaches the total of the dataset in one train run.

* `SHUFFLE_BUFFER_SIZE` **(DEPRECATED)** - was used for the [TensorArray](https://www.tensorflow.org/api_docs/python/tf/TensorArray) to randomly shuffle batches of dataset.

* `EPOCHS` - the value of total re-occurrence of the training on a dataset while updating the previous weights of the model, to refine it to reach optimum accuracy.

* `MAX_SEQ_LENGTH` **(DEPRECATED)** - was used to limit the sequence of inputted data into the ML model.

* `NUM_FEATURES` **(DEPRECATED)** - was used to calculate the size number of features (the points of interest to detect an image in a [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network)).


```Python
###############
# YOLO SET-UP #
###############
# Directories
YOLO_BACKUP = PATH_DATA + "backup/yolo/"
YOLO_FRAMES = PATH_DATA + "frames/"
YOLO_METADATA = PATH_DATA + "metadata/yolo/"

# Files
YOLO_MAIN_DATA = YOLO_METADATA + "main.txt"
YOLO_VALID_DATA = YOLO_METADATA + "valid.txt"
YOLO_TEST_DATA = YOLO_METADATA + "test.txt"
YOLO_DATA_FILE = YOLO_METADATA + "obj.data"
YOLO_NAMES_FILE = YOLO_METADATA + "obj.names"
```
The reason the variables of the YOLO model is separate is because it uses a completely different framework from [Tensorflow](https://www.tensorflow.org/), where it instead uses the [Darknet framework](https://github.com/AlexeyAB/darknet), which the officially supported way to train the YOLO models.

| Variables Meanings |
|--------------------|
* `YOLO_BACKUP` - this is where the checkpoints for the YOLO models trained weights will be stored.

* `YOLO_FRAMES` - this is the place where the extracted dataset video frames are stored with text files that includes the pixel coordinates of the location of the missing person for object detection.

* `YOLO_METADATA` - this is the metadata that allows the run of the YOLO model under the [Darknet framework](https://github.com/AlexeyAB/darknet).

* `YOLO_MAIN_DATA` - text file that links to a subset data of the frames in `YOLO_FRAMES` to be used for training the model.

* `YOLO_VALID_DATA` - text file that links to a subset data of the frames in `YOLO_FRAMES` to be used for validating the model

* `YOLO_TEST_DATA` - text file that links to a subset data of the frames in `YOLO_FRAMES` to be used for testing the model

* `YOLO_DATA_FILE` - text files that includes all the important information of the number of name classes and the paths to the `YOLO_MAIN_DATA`, `YOLO_MAIN_DATA`, `YOLO_NAMES_FILE` (the text file that contain the classes names) and `YOLO_BACKUP`

* `YOLO_NAMES_FILE` - text file that contains the classes names from the array `CLASSES` in the second code snippet above. Each class name is split into new lines, where label 0 corresponds to the first line and label 1 corresponds to the second line.


```Python
#########################
# YOLO Hyper-parameters #
#########################
YOLO_BATCH = 64
YOLO_SUBDIVISION = 32
YOLO_MAX_BATCHES = 4000
YOLO_LOWER_STEPS = 400
YOLO_UPPER_STEPS = 450
```
These are the special constant variables parameters that tweak how the training of YOLO model should run in [Data Manipulation and Training of ML Models Stack](#data-manipulation-and-training-of-ml-models-stack).

| Variables Meanings |
|--------------------|
* `YOLO_BATCH` - the number of batches of dataset images to be run at once with process of training.

* `YOLO_SUBDIVISION` - the division of the batches to be loaded in GPU cores, so when `YOLO_BATCH` is `64` and the `YOLO_SUBDIVISION` is `32`, the GPU will split the batches into 2 halves in parallel to ease the memory on the GPU.

* `YOLO_MAX_BATCHES` - this is how the number of re-occurrence of training per 1 of the sum total of `YOLO_BATCH` (so from the code snippet above, [`64` batch is equal to `1` max batch](https://github.com/AlexeyAB/darknet/wiki/CFG-Parameters-in-the-%5Bnet%5D-section)).

* `YOLO_LOWER_STEPS` and `YOLO_UPPER_STEPS` - the steps that once the `YOLO_MAX_BATCHES` total value reach in the steps boundary, [then the policy will change the current learning rate](https://github.com/AlexeyAB/darknet/wiki/CFG-Parameters-in-the-%5Bnet%5D-section).


# Limitations 

First, the [Data Processing Stack](#data-processing-stack) is solely designed to detect `"Person"` or `"No Person"` in terms of Image Classification models (for, example [ResNet50V2](https://keras.io/api/applications/resnet/#resnet50v2-function)) or only `"Person"` class name will be used in the case of Object Detection models (for example, [Tiny-YOLOv3](https://github.com/AlexeyAB/darknet)).

```Python
if len(contours) != 0:
                    frames_with_missing_person.append(temp_frame)
                    for label in CLASSES:
                        label_name = "Person"
                        if label == label_name:
                            labels_for_missing_person.append(CLASSES.index(label_name))
                            print("Found:", label_name, ", Label Num:", CLASSES.index(label_name))
                    proces_images(path, contours, frame, temp_frame, progress)
                else:
                    frames_without_missing_person.append(temp_frame)
                    for label in CLASSES:
                        label_name = "No Person"
                        if label == label_name:
                            labels_for_without_missing_person.append(CLASSES.index(label_name))
                            print("Found:", label_name, ", Label Num:", CLASSES.index(label_name))
```
As you can see the above code snippet, the conditions are hardcoded to label the dataset as `"Person"` or `"No Person"`.

```Python
if YOLO_MODEL is True:
                file_without_extension = os.path.splitext(os.path.basename(path))[0]
                frame_file_name = "{0}{1}-{2}".format(YOLO_FRAMES, file_without_extension, progress)

                if os.path.isfile(frame_file_name + ".jpg") is False:
                    cv2.imwrite(frame_file_name + ".jpg", clean_frame)

                # Coordinates data
                if os.path.isfile(frame_file_name + ".txt"):
                    coord_file = open(frame_file_name + ".txt", "a")
                    coord_file.write("0 {0} {1} {2} {3}\n".format((x + 10) / IMG_SIZE, (y + 10) / IMG_SIZE,
                                                                  w / IMG_SIZE, h / IMG_SIZE))
                    coord_file.close()
                else:
                    coord_file = open(frame_file_name + ".txt", "w")
                    coord_file.write("0 {0} {1} {2} {3}\n".format((x + 10) / IMG_SIZE, (y + 10) / IMG_SIZE,
                                                                  w / IMG_SIZE, h / IMG_SIZE))
                    coord_file.close()
```
In here, the above code snippet has hardcoded the number `"0"` which corresponds to the class name `"Person"` in the `YOLO_NAMES_FILE` (see above to the code snippet for the [[YOLO SET-UP](#setup-and-hyper-parameters-for-the-program-to-work) in the Usage section](#usage))

Second, there is no constant variable to change the input size of the YOLO model. The model uses the default pixel input size of `416x416` when being read by the [Darknet framework](https://github.com/AlexeyAB/darknet) in the [Data Manipulation and Training of ML Models Stack](#data-manipulation-and-training-of-ml-models-stack).

Third, due to the YOLO model running through the [Darknet framework](https://github.com/AlexeyAB/darknet), which is a framework made in C/C++ and mainly configured through changing values in files, it is hard to produce analytics other than the provided ones from the framework (for example, [mAP](https://towardsdatascience.com/map-mean-average-precision-might-confuse-you-5956f1bfa9e2) with the loss over iteration is provided by the framework). 

However, there is the option to convert the trained YOLO model into [Tensorflow compatible saved model](https://github.com/AlexeyAB/darknet/wiki/Converting-Yolo-v3-models-to-TensorFlow-and-OpenVINO(IR)-models#converting-a-yolo-v3-model-darknet---tensorflow); but due to time constraints, I have not tested the suggested code to convert the model, thus, I do not know if it could work.


Lastly, the [Data Manipulation and Training of ML Models Stack](#data-manipulation-and-training-of-ml-models-stack), and [Analytics for ML Models Stack)](#analytics-for-ml-models-stack) are basically hardcoded like the [Data Processing Stack](#data-processing-stack). Although the parameters in the [Setup and Hyper-parameters for the program to work](#setup-and-hyper-parameters-for-the-program-to-work) can configure the process of the training of ML models, these 2 other stacks cannot automatically load a different ML models by just adding extra variables in the ["Model to Run" boolean code snippet](#setup-and-hyper-parameters-for-the-program-to-work) - you will need to add the new ML models manually in a similar manner of the other ML models. 

It is designed that way because of time constraints and the original focus was solely to test the Tiny-YOLOv3, ResNet50V2 and MobileNetV2 models, which was the best candidates for TCSR use of finding missing persons using drone's imaging data, according to the research done in the given time period of the dissertation.


# Author and Acknowledgment

Author: [Abdullah Alshadadi](https://github.com/Srking501)

Special Thanks to: [The Centre for Search Research (TCSR)](https://tcsr.org.uk/)


# License

...
