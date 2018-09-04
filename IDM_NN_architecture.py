#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Definition of the Neural Network baseline model architecture

@author Guigui

to be loaded with

from IDM_NN_architecture import *

"""

#There are two ways to build Keras models: sequential and functional.
#The sequential API allows you to create models layer-by-layer for most problems.
#It is limited in that it does not allow you to create models that share layers
#or have multiple inputs or outputs. The sequential model is a linear stack of layers.
# see https://keras.io/models/sequential/
# and https://keras.io/getting-started/sequential-model-guide/


from keras.models import Sequential
from keras.layers import Dense


#We then define the baseline model
#def baseline_model():
def baseline_model():
    # create model
    model = Sequential()
    #Layers are added piecewise
    #the first number is the number of neurons and the 2nd the # of input attributes
    model.add(Dense(8, input_dim=5, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(12, kernel_initializer='uniform', activation='relu'))
    #If no activation function is declared, the default one is used (here linear)
    # This is suitable for a regression problem.
    model.add(Dense(6, kernel_initializer='uniform'))
    # Compile model to configure the learning process
    # In principle it can also take a metric as argument but this is only needed
    # for a classification problem, not a regression problem
    # The optimizer is the method by which the NN will minimize the loss function
    # Here we leave the parameters as default
    #see https://keras.io/optimizers/
    # We took the adadelta optimizer since it is the one which gave the best score
    model.compile(loss='mean_squared_error', optimizer='adadelta')
    return model


