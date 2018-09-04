#!/usr/bin/env python3                                               |  ---------------------------------------------------------------------
# -*- coding: utf-8 -*-

"""

loading the RandomForst model

used to fit in the IDM model
"""

#first load the quantile transformation which has been used          |                                                                       
# in the training to normalise the data such that the                |  ---------------------------------------------------------------------
#features and target variables have a gaussian distribution          |  ---------------------------------------------------------------------
# to improve the performance of the ML algorithm                   |  ---------------------------------------------------------------------
# It is only important for transforming the target variables         |  ---------------------------------------------------------------------
# (the cross section values) since the transformation of the         |  ---------------------------------------------------------------------
# features variables is included in the pipeline (see below)         |  ---------------------------------------------------------------------
# !!! WARNING !!! The fit_transform was applied on the               |  ---------------------------------------------------------------------
# whole target variables (8 cross sections), thus,                   |  ---------------------------------------------------------------------
# it has to be applied on an 8-D array.                              |  ---------------------------------------------------------------------
from sklearn.externals import joblib
quantile_transformer = joblib.load('normalizer_rf.save')
print("QuantileTransformation fitted to the target training data (y_train) loaded...")

#Next we load the RandomForest model
#In fact what we load is a pipeline of transformation
# plus the random forest estimator

pipeline = joblib.load('RF_pipeline.gz')


print("Remember that the predictions of the pipeline are the Quantile Transformed cross sections")
print("The Quantile Transformation object is called quantile_transformer")
print("To get the true cross sections you need to untransfrom them through e.g:")
print("predictions = quantile_transformer.inverse_transform(pipeline.predict(X))")
print("each step in the pipeline can be accessed through pipeline.named_steps['name of the estimator']")
print("the 3 steps are 'normalize','standardize' and 'rf'")
