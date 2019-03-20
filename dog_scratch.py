#!/usr/bin/env python3
# coding: utf-8

from glob import glob
import time
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFile
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing import image
from keras.utils import np_utils
from sklearn.datasets import load_files
from tqdm import tqdm

import tictoc

ImageFile.LOAD_TRUNCATED_IMAGES = True


# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

# Models

def udacity_model():
    """Example udacity model for quick results on dog breed classification

    Performance tops out on validation set on epoch 284 at val_acc=15%
    """
    model_name = 'udacity_model'
    # layer                     output shape            params
    #------------------------------------------------------------
    # conv2d(2,2)               (None, 223, 223, 16)    208
    # maxPool2d(2,2)            (None, 111, 111, 16)    0
    # conv2d(2,2)               (None, 110, 110, 32)
    # maxPool2d(2,2)            (None, 55, 55, 32)
    # conv2d(2,2)               (None, 54, 54, 64)
    # maxPool2d(2,2)            (None, 27, 27, 64)
    # globalaveragepooling2d    (None, 64)
    # dense                     (None, 133)
    model = Sequential()
    model.add(Conv2D(16, (2, 2), input_shape=(224, 224, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (2, 2), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(133, activation='softmax'))

    return (model, model_name)


def matt_model1():
    """Like udacity model but using 3x3 kernels

    Performance tops out ??
    """
    model_name = 'matt_model1'
    # layer                     output shape            params
    #------------------------------------------------------------
    # conv2d(2,2)               (None, 223, 223, 16)    208
    # maxPool2d(2,2)            (None, 111, 111, 16)    0
    # conv2d(2,2)               (None, 110, 110, 32)
    # maxPool2d(2,2)            (None, 55, 55, 32)
    # conv2d(2,2)               (None, 54, 54, 64)
    # maxPool2d(2,2)            (None, 27, 27, 64)
    # globalaveragepooling2d    (None, 64)
    # dense                     (None, 133)
    inputs = Input(shape=(224, 224, 3))
    x = Conv2D(16, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((3, 3))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((3, 3))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((3, 3))(x)
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(133, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)

    return (model, model_name)


class MattPlotCallback(Callback):
    def __init__(self, plot_loss=True, plot_error=True):
        self.plot_loss = plot_loss
        self.plot_error = plot_error
    def on_train_begin(self, logs={}):
        # Error fig
        if self.plot_error:
            self.fig_err, self.ax_err = plt.subplots()
            self.ax_err.set_title("Error During Training")
            self.ax_err.set_xlabel("Epoch")
            self.ax_err.set_ylabel("Error (%)")
        # Loss fig
        if self.plot_loss:
            self.fig_loss, self.ax_loss = plt.subplots()
            self.ax_loss.set_title("Loss During Training")
            self.ax_loss.set_xlabel("Epoch")
            self.ax_loss.set_ylabel("Loss")
        if self.plot_loss or self.plot_error:
            plt.ion()
        # data setup
        self.epochs = [None,]
        self.acc = [None,]
        self.val_acc = [None,]
        self.loss = [None,]
        self.val_loss = [None,]

    def on_epoch_end(self, epoch, logs={}):
        #print(logs)
        # TODO: add legends to plots
        self.epochs.append(epoch)
        if self.plot_error:
            if 'acc' in logs:
                self.acc.append(logs['acc'])
                self.ax_err.plot(self.epochs[-2:], self.acc[-2:],'ro-')
            if 'val_acc' in logs:
                self.val_acc.append(logs['val_acc'])
                self.ax_err.plot(self.epochs[-2:], self.val_acc[-2:],'bo-')
        if self.plot_loss:
            if 'loss' in logs:
                self.loss.append(logs['loss'])
                self.ax_loss.plot(self.epochs[-2:], self.acc[-2:],'ro-')
            if 'val_loss' in logs:
                self.val_loss.append(logs['val_loss'])
                self.ax_loss.plot(self.epochs[-2:], self.val_acc[-2:],'bo-')
        if self.plot_loss or self.plot_error:
            plt.pause(0.001)


# Data Setup ---------------------
# load train, test, and validation datasets
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

if os.path.isfile('train_tensors.npy') and os.path.isfile('valid_tensors.npy') and os.path.isfile('test_tensors.npy'):
    print("Loading train_tensors...")
    train_tensors = np.load('train_tensors.npy')
    print("Loading valid_tensors...")
    valid_tensors = np.load('valid_tensors.npy')
    print("Loading test_tensors...")
    test_tensors = np.load('test_tensors.npy')

else:
    # load list of dog names
    dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

    # print statistics about the dataset
    print('There are %d total dog categories.' % len(dog_names))
    print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
    print('There are %d training dog images.' % len(train_files))
    print('There are %d validation dog images.' % len(valid_files))
    print('There are %d test dog images.'% len(test_files))

    # pre-process the data for Keras
    train_tensors = paths_to_tensor(train_files).astype('float32')/255
    valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
    test_tensors = paths_to_tensor(test_files).astype('float32')/255

    # save the resulting numpy ndarrays
    np.save('train_tensors', train_tensors)
    np.save('valid_tensors', valid_tensors)
    np.save('test_tensors', test_tensors)


# Training Setup -----------------
# get model
(model, model_name) = udacity_model()
#(model, model_name) = matt_model1()
model.summary()
# number of epochs to use to train the model.
epochs = 2


# housekeeping
model_save_dir = 'saved_models_' + model_name
history_save_file = 'train_history_' + model_name + '.json'
os.makedirs(model_save_dir, exist_ok=True)

# Compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Instantiate callbacks
checkpointer = ModelCheckpoint(
        filepath=model_save_dir + '/weights.best.from_scratch.hdf5',
        verbose=1, save_best_only=True
        )
checkpointer2 = ModelCheckpoint(
        filepath=model_save_dir + '/weights.epoch{epoch:04d}.hdf5',
        verbose=0, period=20
        )
mattplot_callback = MattPlotCallback()

history = {}
mytimer = tictoc.Timer()
mytimer.start()

# Train the model
try:
    history = model.fit(
            train_tensors, train_targets,
            validation_data=(valid_tensors, valid_targets),
            epochs=epochs,
            batch_size=20,
            callbacks=[checkpointer, checkpointer2, mattplot_callback],
            verbose=1
            )
except KeyboardInterrupt:
    print("Stopped prematurely via keyboard interrupt.")
mytimer.eltime_pr("training time: ")

# save final model+weights
print(history.history)
model.save(model_save_dir + '/weights.final.hd5')

# save history to json file
with open(history_save_file, "w") as train_hist_fh:
    json.dump(history.history, train_hist_fh)

# plot accuracy vs. epoch from history
fig, ax = plt.subplots()
# ax belongs to fig
ax.set_title("Training Error")
ax.set_xlabel("Epoch")
ax.set_ylabel("Error (%)")
if 'acc' in history.history:
    ax.plot(100*(1-np.array(history.history['acc'])),'b')
if 'val_acc' in history.history:
    ax.plot(100*(1-np.array(history.history['val_acc'])),'r')
plt.show()

# Load the model with the best validation loss
model.load_weights('saved_models/weights.best.from_scratch.hdf5')

# Try out your model on the test dataset of dog images.
# Ensure that your test accuracy is greater than 1%.

# get index of predicted dog breed for each image in test set
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

print("exit(0)")
exit(0)


# ---
# <a id='step4'></a>
# ## Step 4: Use a CNN to Classify Dog Breeds
#
# To reduce training time without sacrificing accuracy, we show you how to train a CNN using transfer learning.  In the following step, you will get a chance to use transfer learning to train your own CNN.
#
# ### Obtain Bottleneck Features

# In[ ]:


bottleneck_features = np.load('bottleneck_features/DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']


# ### Model Architecture
#
# The model uses the the pre-trained VGG-16 model as a fixed feature extractor, where the last convolutional output of VGG-16 is fed as input to our model.  We only add a global average pooling layer and a fully connected layer, where the latter contains one node for each dog category and is equipped with a softmax.

# In[ ]:


VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))

VGG16_model.summary()


# ### Compile the Model

# In[ ]:


VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# ### Train the Model

# In[ ]:


checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5',
                               verbose=1, save_best_only=True)

VGG16_model.fit(train_VGG16, train_targets,
          validation_data=(valid_VGG16, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)


# ### Load the Model with the Best Validation Loss

# In[ ]:


VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')


# ### Test the Model
#
# Now, we can use the CNN to test how well it identifies breed within our test dataset of dog images.  We print the test accuracy below.

# In[ ]:


# get index of predicted dog breed for each image in test set
VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]

# report test accuracy
test_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


# ### Predict Dog Breed with the Model

# In[ ]:


from extract_bottleneck_features import *

def VGG16_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_VGG16(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = VGG16_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


# ---
# <a id='step5'></a>
# ## Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)
#
# You will now use transfer learning to create a CNN that can identify dog breed from images.  Your CNN must attain at least 60% accuracy on the test set.
#
# In Step 4, we used transfer learning to create a CNN using VGG-16 bottleneck features.  In this section, you must use the bottleneck features from a different pre-trained model.  To make things easier for you, we have pre-computed the features for all of the networks that are currently available in Keras:
# - [VGG-19](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz) bottleneck features
# - [ResNet-50](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) bottleneck features
# - [Inception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz) bottleneck features
# - [Xception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz) bottleneck features
#
# The files are encoded as such:
#
#     Dog{network}Data.npz
#
# where `{network}`, in the above filename, can be one of `VGG19`, `Resnet50`, `InceptionV3`, or `Xception`.  Pick one of the above architectures, download the corresponding bottleneck features, and store the downloaded file in the `bottleneck_features/` folder in the repository.
#
# ### (IMPLEMENTATION) Obtain Bottleneck Features
#
# In the code block below, extract the bottleneck features corresponding to the train, test, and validation sets by running the following:
#
#     bottleneck_features = np.load('bottleneck_features/Dog{network}Data.npz')
#     train_{network} = bottleneck_features['train']
#     valid_{network} = bottleneck_features['valid']
#     test_{network} = bottleneck_features['test']

# In[ ]:


### TODO: Obtain bottleneck features from another pre-trained CNN.


# ### (IMPLEMENTATION) Model Architecture
#
# Create a CNN to classify dog breed.  At the end of your code cell block, summarize the layers of your model by executing the line:
#
#         <your model's name>.summary()
#
# __Question 5:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  Describe why you think the architecture is suitable for the current problem.
#
# __Answer:__
#
#

# In[ ]:


### TODO: Define your architecture.


# ### (IMPLEMENTATION) Compile the Model

# In[ ]:


### TODO: Compile the model.


# ### (IMPLEMENTATION) Train the Model
#
# Train your model in the code cell below.  Use model checkpointing to save the model that attains the best validation loss.
#
# You are welcome to [augment the training data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html), but this is not a requirement.

# In[ ]:


### TODO: Train the model.


# ### (IMPLEMENTATION) Load the Model with the Best Validation Loss

# In[ ]:


### TODO: Load the model weights with the best validation loss.


# ### (IMPLEMENTATION) Test the Model
#
# Try out your model on the test dataset of dog images. Ensure that your test accuracy is greater than 60%.

# In[ ]:


### TODO: Calculate classification accuracy on the test dataset.


# ### (IMPLEMENTATION) Predict Dog Breed with the Model
#
# Write a function that takes an image path as input and returns the dog breed (`Affenpinscher`, `Afghan_hound`, etc) that is predicted by your model.
#
# Similar to the analogous function in Step 5, your function should have three steps:
# 1. Extract the bottleneck features corresponding to the chosen CNN model.
# 2. Supply the bottleneck features as input to the model to return the predicted vector.  Note that the argmax of this prediction vector gives the index of the predicted dog breed.
# 3. Use the `dog_names` array defined in Step 0 of this notebook to return the corresponding breed.
#
# The functions to extract the bottleneck features can be found in `extract_bottleneck_features.py`, and they have been imported in an earlier code cell.  To obtain the bottleneck features corresponding to your chosen CNN architecture, you need to use the function
#
#     extract_{network}
#
# where `{network}`, in the above filename, should be one of `VGG19`, `Resnet50`, `InceptionV3`, or `Xception`.

# In[ ]:


### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.


# ---
# <a id='step6'></a>
# ## Step 6: Write your Algorithm
#
# Write an algorithm that accepts a file path to an image and first determines whether the image contains a human, dog, or neither.  Then,
# - if a __dog__ is detected in the image, return the predicted breed.
# - if a __human__ is detected in the image, return the resembling dog breed.
# - if __neither__ is detected in the image, provide output that indicates an error.
#
# You are welcome to write your own functions for detecting humans and dogs in images, but feel free to use the `face_detector` and `dog_detector` functions developed above.  You are __required__ to use your CNN from Step 5 to predict dog breed.
#
# Some sample output for our algorithm is provided below, but feel free to design your own user experience!
#
# ![Sample Human Output](images/sample_human_output.png)
#
#
# ### (IMPLEMENTATION) Write your Algorithm

# In[ ]:


### TODO: Write your algorithm.
### Feel free to use as many code cells as needed.


# ---
# <a id='step7'></a>
# ## Step 7: Test Your Algorithm
#
# In this section, you will take your new algorithm for a spin!  What kind of dog does the algorithm think that __you__ look like?  If you have a dog, does it predict your dog's breed accurately?  If you have a cat, does it mistakenly think that your cat is a dog?
#
# ### (IMPLEMENTATION) Test Your Algorithm on Sample Images!
#
# Test your algorithm at least six images on your computer.  Feel free to use any images you like.  Use at least two human and two dog images.
#
# __Question 6:__ Is the output better than you expected :) ?  Or worse :( ?  Provide at least three possible points of improvement for your algorithm.
#
# __Answer:__

# In[ ]:


## TODO: Execute your algorithm from Step 6 on
## at least 6 images on your computer.
## Feel free to use as many code cells as needed.

