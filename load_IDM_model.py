#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

loading the Neural Network model for predicting new results

in the IDM model
"""

#first load the quantile transformation which has been used
# in the training to normalise the data such that the 
#features and target variables have a gaussian distribution
# to improve the performance of the neural network
# It is only important for transforming the target variables
# (the cross section values) since the transformation of the
# features variables is included in the pipeline (see below)
# !!! WARNING !!! The fit_transform was applied on the
# whole target variables (8 cross sections), thus,
# it has to be applied on an 8-D array.
from sklearn.externals import joblib

quantile_transformer = joblib.load('normalizer.save')

#Next load the NN model. First the architecture

from IDM_NN_architecture import *

# Then the pipeline that was used to
#   - normalise the features (input variables) with QuantileTransformer
#   - scale the features with StandardScaler
#   - fit the model with the KerasRegressor

from keras.models import load_model 

pipeline = joblib.load('keras_IDM_model.pkl')
pipeline.named_steps['mlp'].model = load_model('keras_IDM_model.h5')


