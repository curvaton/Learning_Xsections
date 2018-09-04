#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 09:58:02 2018

@author: Guigui

Regression with the Keras Deep Learning Library in Python

Steps mostly followed from:
https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/    
"""

#The data is loaded from the file load_IDM_data.py
# This is to ease the loading of the data when the 
# saved model (see below) will be reused
from load_IDM_data import *

#To ease the fitting with the NN and in particular avoid negative predicted
#cross sections (that we had at the beginning of the project),
#the data should look as gaussian as possible. One way of doing
# it is through the QuantileTransformer library (a non-linear transformation on 
# the data), see http://scikit-learn.org/stable/modules/preprocessing.html
# Later we will use Pipeline to chain the transformation. However, pipelines
# will always pass y through unchanged, we have to do the transformation outside
# the pipeline. That we do below. This will be annoying for later use, for example
# using the saved weights and model to predict new data, since it will predict
# transformed data, that we will have to inverse transform. 
from sklearn.preprocessing import QuantileTransformer
quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=0)

y_train_norm = quantile_transformer.fit_transform(y_train)
y_test_norm  = quantile_transformer.transform(y_test)

#For later use of predicting new data, we have to use the very same
#transformation on it to avoid overfitting. We thus need to save this
# transformation into a file tha we will need to load to predict new
#data

from sklearn.externals import joblib
quantile_transformer_filename = "normalizer.save"
joblib.dump(quantile_transformer,quantile_transformer_filename)


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

#Import the NN architecture
from IDM_NN_architecture import *

# fix random seed for reproducibility
seed = 123
np.random.seed(seed)


#Now let us define the regression estimator

from keras.wrappers.scikit_learn import KerasRegressor

# evaluate model with standardized dataset
# This is the implementation with fixed number of epochs and batch_size
regressor = KerasRegressor(build_fn=baseline_model, epochs=150, batch_size=50, verbose=0)


#Now we will perform the standardisation (scaling) of the dataset during the model
#evaluation process, within each fold of the cross validation.
#To perform this step one can use the scikit-learn Pipeline framework.
# Pipelines work by allowing for a linear sequence of data transforms to be chained
# together culminating in a modeling process that can be evaluated.
# see http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
# and http://scikit-learn.org/stable/modules/pipeline.html#pipeline
# The role of pipeline: it is better to standardise the dataset at each fold
# instead of standardising the whole dataset first. Since for standardisation
# the mean and variance is computed, if we first apply it on the whole dataset and 
# then split it into different folds, each fold contains somehow information about 
# the whole dataset, some information has leaked. Indeed, the folds will not be 
# centered on their own mean, but the mean of the whole dataset. Thus, it is better 
# to first split the dataset into k-folds, and then standardise each fold.

from sklearn.pipeline import Pipeline

# evaluate model with normalized and standardized dataset

estimators = []
estimators.append(('normalize', QuantileTransformer(output_distribution='normal', random_state=0)))
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', regressor))
pipeline = Pipeline(estimators)



#The final step is to evaluate this baseline model. 
#We will use 10-fold cross validation to evaluate the model.
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold


kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X_train, y_train_norm, cv=kfold)
print("Standardized: %.15f (%.15f) MSE" % (results.mean(), results.std()))

#predictions_train_trans = cross_val_predict(pipeline, X_train, y_train_norm, cv=kfold)

#Convert into a 2D array
#predictions_train_trans = predictions_train_trans.reshape(-1,1)
#predictions_train = quantile_transformer.inverse_transform(predictions_train_trans) 

#To be able to save the model and evaluate it
#we have to fit again because cross val does not store the fit parameters
pipeline.fit(X_train,y_train_norm)
predictions_train_new = quantile_transformer.inverse_transform(pipeline.predict(X_train))
predictions_test = quantile_transformer.inverse_transform(pipeline.predict(X_test))

#Score of the regression
from sklearn.metrics import mean_squared_error
print("MSE train : %.15f " % mean_squared_error(y_train,predictions_train_new))
print("MSE test : %.15f " % mean_squared_error(y_test,predictions_test))

with open('Results.txt','a+',newline='\n') as f:
    f.write('Results for 1 layer with 13 neurons \n')
    f.write('Standardized: %.15f (%.15f) MSEi \n'% (results.mean(), results.std()))
    f.write('MSE train : %.15f \n' % mean_squared_error(y_train,predictions_train_new))
    f.write('MSE test : %.15f \n' % mean_squared_error(y_test,predictions_test))

f.close()
#Saving the Keras model
from keras.models import load_model

pipeline.named_steps['mlp'].model.save('keras_IDM_model.h5')

#This hack allows to save the sklearn pipeline 
pipeline.named_steps['mlp'].model = None

#Finally save the pipeline:
joblib.dump(pipeline,'keras_IDM_model.pkl')

del pipeline
"""

# Scoring
#score = pipeline.score(X_test, y_test_norm)
#print("score on y_test: %.15f" % score)

#Things to think about:
# Do I need to split anyway into a training set and a test set ?
# Use of cross_val_predict() ? => done
# Saving and storing the model => done
# Are the weights only for standardised sets ? i.e if I save the model
# do I need to standardise explicitly the test set ? => Yes
# Direct implementation of the model ? (Without going through Sequential)
# Why is the score different from cross_val and pipeline.score ?
"""
