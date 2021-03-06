# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 00:20:33 2018

@author: Sanjith Hebbar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Dropout, Input
from scipy.misc import toimage
from keras import backend as K
from keras.callbacks import LearningRateScheduler
K.set_image_dim_ordering("tf")

#Initialise random seed for replicating results
np.random.seed(0)

# Load the dataset
(X_train, y_train),(X_test, y_test) = cifar10.load_data()

# Preprocess data
X_train = X_train.astype('float')
X_test = X_test.astype('float')
X_train = X_train / 255
X_test = X_test / 255

# Encode Target Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
onehotencoder = OneHotEncoder(categorical_features = [0])
y_train[:,0] = labelencoder.fit_transform(y_train[:,0])
y_train = onehotencoder.fit_transform(y_train).toarray()
y_test[:,0] = labelencoder.fit_transform(y_test[:,0])
y_test = onehotencoder.fit_transform(y_test).toarray()

# Import Pretrained Model
from keras.applications.vgg16 import VGG16
classifier = VGG16(include_top = False, weights = 'imagenet')
    
#Create input layer
input_layer = Input(shape = (32,32,3))

#Use the generated model 
new_vgg16 = classifier(inputs = input_layer)
   
# Adding the connected layers
x = Flatten()(new_vgg16)
x = Dense(10, activation = 'softmax')(x)

# Creating Custom Model
new_classifier = Model(inputs = input_layer, outputs = x)
new_classifier.summary()

# Compiling model and using SGD optimizer
from keras.optimizers import SGD
epochs = 10
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr = lrate, momentum = 0.9, decay = decay, nesterov = False)
new_classifier.compile(optimizer = sgd, metrics = ['accuracy'], loss = 'categorical_crossentropy')

# Adding Image Augumentation
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(zoom_range = 0.2,
                             horizontal_flip = True)

# Train the model
cnn = new_classifier.fit_generator(datagen.flow(X_train, y_train, batch_size = 32),
                               steps_per_epoch = (X_train.shape[0])/32, epochs = epochs, 
                               validation_data = (X_test, y_test))

#cnn = new_classifier.fit(X_train, y_train, validation_data = (X_test,y_test), epochs=epochs, batch_size=32)

# Final evaluation of the model
scores = new_classifier.evaluate(X_test, y_test, verbose = 0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Save Model
new_classifier.save("CNNmodelPretrained.h5")

# Load Model if already saved
# new_classifier = load_model("CNNmodelPretrained.h5")

# Visualise results
# Accuracy
plt.figure(figsize = (6,6))
plt.plot(cnn.history['acc'],'blue')
plt.plot(cnn.history['val_acc'],'red')
plt.xlabel("Number of Epochs")
plt.xticks(np.arange(0, epochs+1, epochs/10))
plt.ylabel("Accuracy")
plt.yticks(np.arange(0.6, 1, 0.05))
plt.title("Training Accuracy vs Test Accuracy")
plt.legend(['Training','Testing'])

# Loss
plt.figure(figsize = (6,6))
plt.plot(cnn.history['loss'],'blue')
plt.plot(cnn.history['val_loss'],'red')
plt.xlabel("Number of Epochs")
plt.xticks(np.arange(0, epochs+1, epochs/10))
plt.ylabel("Loss")
plt.title("Training Loss vs Test Loss")
plt.legend(['Training','Testing'])