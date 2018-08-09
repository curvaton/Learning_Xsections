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
currentdir = os.getcwd()
os.chdir(currentdir)

#Listing the directory content
#os.listdir(os.getcwd())

#importing the relevant libraries
import numpy as np
import pandas as pd





#loading the dataset

dataset = pd.read_csv('IDM_xsecs_13TeV.csv')

#Removing data not meeting some criteria from the dataset
dataset = dataset.ix[~(dataset['xsec_3536_13TeV'] < 0.001)]

#Splitting the dataset into the feature variables (the model parameters MH0,MA0,
#MHC, lamL, lam2) and the dependent variables (xsec_###)

#Feature set
X = dataset.iloc[:,0:5].values

#Target set
y = dataset.iloc[:,5:13].values


#Splitting the dataset into a Training set and a Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

#For the moment splitting the target set for each production cross section
# We use reshape to convert to 2D arrays because the API's expect so
y_3535_train = y_train[:,0].reshape(-1,1)
y_3636_train = y_train[:,1].reshape(-1,1)
y_3737_train = y_train[:,2].reshape(-1,1)
y_3537_train = y_train[:,3].reshape(-1,1)
y_3637_train = y_train[:,4].reshape(-1,1)
y_3735_train = y_train[:,5].reshape(-1,1)
y_3736_train = y_train[:,6].reshape(-1,1)
y_3536_train = y_train[:,7].reshape(-1,1)

y_3535_test = y_test[:,0].reshape(-1,1)
y_3636_test = y_test[:,1].reshape(-1,1)
y_3737_test = y_test[:,2].reshape(-1,1)
y_3537_test = y_test[:,3].reshape(-1,1)
y_3637_test = y_test[:,4].reshape(-1,1)
y_3735_test = y_test[:,5].reshape(-1,1)
y_3736_test = y_test[:,6].reshape(-1,1)
y_3536_test = y_test[:,7].reshape(-1,1)

#To ease the fittig with the NN and in particular avoid negative predicted
#cross sections, the data should look as gaussian as possible. One way of doing
# it is through the QuantileTransformer library (a non-linear transformation on 
# the data), see http://scikit-learn.org/stable/modules/preprocessing.html
from sklearn.preprocessing import QuantileTransformer
quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=0)
y_3536_train_trans = quantile_transformer.fit_transform(y_3536_train)
y_3536_test_trans = quantile_transformer.transform(y_3536_test)



#We will need to standardise (rescale) the dataset since we the features and the 
#targets have their own scales can vary wildly within. By standardising the input
#we will gain in model performance. To standardise we use the StandardScaler from
#sklearn

#from sklearn.preprocessing import StandardScaler
#Maybe better to use MinMaxScaler since all values are then
# in the range (0,1) since we vary for the xsec around
#several orders of magnitude

#from sklearn.preprocessing import MinMaxScaler

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
	model.add(Dense(10, input_dim=5, kernel_initializer='normal', activation='relu'))
        model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    #If not activation function is declared, the default one is used (here linear)
    # This is suitable for a regression problem.
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model to configure the learning process
    # In principle it can also take a metric as argument but this is only needed
    # for a classification problem, not a regression problem
    # The optimizer is the method by which the NN will minimize the loss function
    # Here we leave the parameters as default 
    #see https://keras.io/optimizers/
	model.compile(loss='mean_squared_error', optimizer='Nadam')
	return model

# fix random seed for reproducibility
seed = 123
numpy.random.seed(seed)


#Now let us define the regression estimator

from keras.wrappers.scikit_learn import KerasRegressor

# evaluate model with standardized dataset
regressor = KerasRegressor(build_fn=baseline_model, epochs=150, batch_size=50, verbose=0)

#Now we will perform the standardisation (scaling) of the dataset during the model
#evaluation process, within each fold of the cross validation.
#To perform this step one can use the scikit-learn Pipeline framework.
# Pipelines work by allowing for a linear sequence of data transforms to be chained
# together culminating in a modeling process that can be evaluated.
# see http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
# The role of pipeline: it is better to standardise the dataset at each fold
# instead of standardising the whole dataset first. Since for standardisation
# the mean and variance is computed, if we first apply it on the whole dataset and 
# then split it into different folds, each fold contains somehow information about 
# the whole dataset, some information has leaked. Indeed, the folds will not be 
# centered on their own mean, but the mean of the whole dataset. Thus, it is better 
# to first split the dataset into k-folds, and then standardise each fold.

from sklearn.pipeline import Pipeline

# evaluate model with standardized dataset

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', regressor))
pipeline = Pipeline(estimators)



#The final step is to evaluate this baseline model. 
#We will use 10-fold cross validation to evaluate the model.
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold


kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X_train, y_3536_train_trans, cv=kfold)
predictions_train_trans = cross_val_predict(pipeline, X_train, y_3536_train_trans, cv=kfold)
print("Standardized: %.15f (%.15f) MSE" % (results.mean(), results.std()))

#Convert into a 2D array
predictions_train_trans = predictions_train_trans.reshape(-1,1)
predictions_train = quantile_transformer.inverse_transform(predictions_train_trans) 

#To be able to save the model and evaluate it
#we have to fit again because cross val does not store the fit parameters
pipeline.fit(X_train,y_3536_train_trans)
predictions_train_new = quantile_transformer.inverse_transform(pipeline.predict(X_train))
predictions_test = quantile_transformer.inverse_transform(pipeline.predict(X_test))



#Saving the Keras model
from keras.models import load_model
from sklearn.externals import joblib

pipeline.named_steps['mlp'].model.save('keras_IDM_model.h5')

#This hack allows to save the sklearn pipeline 
pipeline.named_steps['mlp'].model = None

#Finally save the pipeline:
joblib.dump(pipeline,'keras_IDM_model.pkl')

del pipeline


#For the moment the training is done over the whole dataset
#This is to make predictions to compare against the "true" data
#pipeline.fit(X_train, y_3536_train)
#predictions = pipeline.predict(X_test)
#score = pipeline.score(predictions, y_3536_test)
#print("score on y_3536_test: %.15f" % score)

#Things to think about:
# Do I need to split anyway into a training set and a test set ?
# Use of cross_val_predict() ?
# Saving and storing the model
# A good cross validation score should be high, at present it is very low
# The score between the training and test sets should be close to each other.
# Dealing with many outputs ? Just change the output layer with 8 nodes
# Destandardise sets ?
# Are the weights only for standardised sets ? i.e if I save the model
# do I need to standardise explicitly the test set ?

