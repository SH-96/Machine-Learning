# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 18:44:16 2018

@author: Sanjith
"""
#Iris Flower Classification

#import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import confusion_metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout
from datetime import datetime
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier

#Neural Network Model
def annModel():
    model = Sequential()
    model.add(Dense(10,input_dim=4,activation='relu'))
    model.add(Dense(3,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer = "adam", metrics = ['accuracy'])
    return model

#import Dataset
dataset = pd.read_csv("iris.csv", header=None)
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

#encode labels
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
onehotencoder_Y = OneHotEncoder(categorical_features = [0])
Y = Y.reshape(-1,1)
Y = onehotencoder_Y.fit_transform(Y).toarray()

#split the dataset
seed = 7
np.random.seed(seed)
kfold=KFold(n_splits=10,shuffle=True,random_state=seed)

#training  & testing the model
print("Training the network...")
model = annModel()
time1=datetime.now()
estimator= KerasClassifier(build_fn=annModel,epochs=200,batch_size=10,verbose=1)
print("Testing the network...")
results=cross_val_score(estimator, X,Y,cv=kfold)
time2=datetime.now()

#results
print("Testing complete.")
print("ACCURACY = ",results.mean()*100)
print("\nTime Taken : ",time2-time1)