# -*- coding: utf-8 -*-
"""
Created on Sat May  9 17:59:00 2020

@author: Atrijo
"""
# Importing the libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initializing he CNN
classifier = Sequential()

# Making a convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))  #Since we are using tensorflow backend

# Step 2- Pooling step
classifier.add(MaxPooling2D(pool_size = (2, 1)))

#Step 3- Flattening
classifier.add(Flatten())

# Full connection- step 4
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Image preprocessing step

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('C:/Users/Atrijo/Desktop/CNN/dataset/training_set',
                                                  target_size=(64, 64),
                                                  batch_size=32,
                                                  class_mode='binary')
test_set = test_datagen.flow_from_directory('C:/Users/Atrijo/Desktop/CNN/dataset/test_set',
                                             target_size=(64, 64),
                                             batch_size=32,
                                             class_mode='binary')
classifier.fit(training_set,
          steps_per_epoch=8000,
          epochs=25,
          validation_data=test_set,
          validation_steps = 2000)