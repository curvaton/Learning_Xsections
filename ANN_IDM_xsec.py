#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 09:58:02 2018

@author: Guigui

Regression with the Keras Deep Learning Library in Python

Steps mostly followed from:
https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/    
"""

import os
os.chdir('/Users/Guigui/Dropbox/Work_LPSC/Sabine_project/Machine_Learning/IDM_xsec')

#Listing the directory content
#os.listdir(os.getcwd())

#importing the relevant libraries
import numpy
import pandas as pd





#loading the dataset

dataset = pd.read_csv('IDM_xsecs_13TeV.csv')


#Splitting the dataset into the feature variables (the model parameters MH0,MA0,
#MHC, lamL, lam2) and the dependent variables (xsec_###)

#Feature set
X = dataset.iloc[:,0:5].values

#Target set
y_3535 = dataset.iloc[:,5].values
y_3636 = dataset.iloc[:,6].values
y_3737 = dataset.iloc[:,7].values
y_3537 = dataset.iloc[:,8].values
y_3637 = dataset.iloc[:,9].values
y_3735 = dataset.iloc[:,10].values
y_3736 = dataset.iloc[:,11].values
y_3536 = dataset.iloc[:,12].values


#We will need to standardise (rescale) the dataset since we the features and the 
#targets have their own scales can vary wildly within. By standardising the input
#we will gain in model performance. To standardise we use the StandardScaler from
#sklearn

from sklearn.preprocessing import StandardScaler

#Next we create our NN model using the Keras libraries

#This is just to check which version of tensorflow is installed
#import tensorflow as tf
#tf.__version__

#There are two ways to build Keras models: sequential and functional.
#The sequential API allows you to create models layer-by-layer for most problems. 
#It is limited in that it does not allow you to create models that share layers 
#or have multiple inputs or outputs. The sequential model is a linear stack of layers.
#Since for our regression purpose we only need one hidden layer,
#loading the sequential model is enough for our purposes
# see https://keras.io/models/sequential/ 
# and https://keras.io/getting-started/sequential-model-guide/
from keras.models import Sequential
from keras.layers import Dense

#We then define the baseline model
def baseline_model():
	# create model
	model = Sequential()
    #Layers are added piecewise
    #the first number is the number of neurons and the 2nd the # of input attributes
	model.add(Dense(5, input_dim=5, kernel_initializer='normal', activation='relu'))
    #If not activation function is declared, the default one is used (here linear)
    # This is suitable for a regression problem.
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model to configure the learning process
    # In principle it can also take a metric as argument but this is only needed
    # for a classification problem, not a regression problem
    # The optimizer is the method by which the NN will minimize the loss function
    # Here we leave the parameters as default 
    #see https://keras.io/optimizers/
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

# fix random seed for reproducibility
seed = 123
numpy.random.seed(seed)


#Now let us define the regression estimator

from keras.wrappers.scikit_learn import KerasRegressor

# evaluate model with standardized dataset
regressor = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=50, verbose=0)

#Now we will perform the standardisation (scaling) of the dataset during the model
#evaluation process, within each fold of the cross validation.
#To perform this step one can use the scikit-learn Pipeline framework.
# see http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

from sklearn.pipeline import Pipeline

# evaluate model with standardized dataset

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', regressor))
pipeline = Pipeline(estimators)
#For the moment the training is done over the whole dataset
pipeline.fit(X, y_3536)
predictions = pipeline.predict(X)

#The final step is to evaluate this baseline model. 
#We will use 10-fold cross validation to evaluate the model.
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, y_3536, cv=kfold)
print("Standardized: %.15f (%.15f) MSE" % (results.mean(), results.std()))




