# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 22:37:15 2018

@author: Sanjith
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 15:43:12 2018
@author: Sanjith Hebbar
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Dropout
from scipy.misc import toimage
from keras import backend as K
K.set_image_dim_ordering("tf")

#Initialise random seed for replicating results
np.random.seed(0)

# Load the dataset
(X_train, y_train),(X_test, y_test) = mnist.load_data()

# Preprocess data
X_train = X_train.astype('float')
X_test = X_test.astype('float')
X_train = X_train / 255
X_test = X_test / 255

X_train = X_train.reshape(len(X_train),28,28,1)
X_test = X_test.reshape(len(X_test),28,28,1)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

# Encode Target Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
onehotencoder = OneHotEncoder(categorical_features = [0])
y_train[:,0] = labelencoder.fit_transform(y_train[:,0])
y_train = onehotencoder.fit_transform(y_train).toarray()
y_test[:,0] = labelencoder.fit_transform(y_test[:,0])
y_test = onehotencoder.fit_transform(y_test).toarray()

# Initialise the classifier
classifier = Sequential()
classifier.add(Conv2D(32,(3,3), input_shape = (28,28,1), padding = 'same', activation = 'relu', data_format = "channels_last"))
classifier.add(Conv2D(32,(3,3), padding = 'same', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.2))

classifier.add(Conv2D(64,(3,3), padding = 'same', activation = 'relu'))
classifier.add(Conv2D(64,(3,3), padding = 'same', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.2))

classifier.add(Conv2D(128,(3,3), padding = 'same', activation = 'relu'))
classifier.add(Conv2D(128,(3,3), padding = 'same', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.2))

classifier.add(Flatten())
classifier.add(Dense(units = 512, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 10, activation = 'softmax'))

# Compile the model with Optimizer
from keras.optimizers import SGD
epochs = 50
lrate = 0.005
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
classifier.compile(optimizer = sgd, metrics = ['accuracy'], loss = 'categorical_crossentropy')
print(classifier.summary())

# Adding Image Augumentation
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(zoom_range=0.2, horizontal_flip=True)

# Train the model
cnn = classifier.fit_generator(datagen.flow(X_train, y_train, batch_size = 32),
                               steps_per_epoch = (X_train.shape[0])/32, epochs = epochs, 
                               validation_data = (X_test, y_test))

"""
Train Model without Augumentation
cnn = classifier.fit(X_train, y_train, validation_data = (X_test,y_test), epochs=epochs, batch_size=32)
"""

# Final evaluation of the model
scores = classifier.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Save Model
classifier.save("CNNmodel.h5")

# Load Model if already saved
# classifier = load_model("CNNmodel.h5")

# Visualise results
# Accuracy
plt.figure(figsize=(6,6))
plt.plot(cnn.history['val_acc'],'red')
plt.xlabel("Number of Epochs")
plt.xticks(np.arange(0,epochs+1,2))
plt.ylabel("Accuracy")
plt.yticks(np.arange(0.97,1.00,0.005))
plt.title("Training Accuracy vs Test Accuracy")
plt.legend(['Training','Testing'])

# Loss
plt.figure(figsize=(6,6))
plt.plot(cnn.history['loss'],'blue')
plt.plot(cnn.history['val_loss'],'red')
plt.xlabel("Number of Epochs")
plt.xticks(np.arange(0,epochs+1,2))
plt.ylabel("Loss")
plt.title("Training Loss vs Test Loss")
plt.legend(['Training','Testing'])