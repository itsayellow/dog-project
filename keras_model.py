#!/usr/bin/env python3

from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential, Model

# Example network from the Udacity problem set
#
# layer                     output shape            params
#-----------------------------------------------------------
# conv2d                    (None, 223, 223, 16)    208
# maxPool2d                 (None, 111, 111, 16)    0
# conv2d                    (None, 110, 110, 32)
# maxPool2d                 (None, 55, 55, 32)
# conv2d                    (None, 54, 54, 64)
# maxPool2d                 (None, 27, 27, 64)
# globalaveragepooling2d    (None, 64)
# dense                     (None, 133)
#-----------------------------------------------------------
# Total params: 19,189.0
# Trainable params: 19,189.0
# Non-trainable params: 0.0


# Sequential Implementation

model1 = Sequential()
model1.add(Conv2D(16, (2, 2), input_shape=(224, 224, 3), activation='relu'))
model1.add(MaxPooling2D((2, 2)))
model1.add(Conv2D(32, (2, 2), activation='relu'))
model1.add(MaxPooling2D((2, 2)))
model1.add(Conv2D(64, (2, 2), activation='relu'))
model1.add(MaxPooling2D((2, 2)))
model1.add(GlobalAveragePooling2D())
model1.add(Dense(133, activation='softmax'))

# print summary of model1
model1.summary()



# Functional Implementation

inputs = Input(shape=(224, 224, 3))
x = Conv2D(16, (2, 2), activation='relu')(inputs)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(32, (2, 2), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (2, 2), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = GlobalAveragePooling2D()(x)
predictions = Dense(133, activation='softmax')(x)

model2 = Model(inputs=inputs, outputs=predictions)
# print summary of model2
model2.summary()
