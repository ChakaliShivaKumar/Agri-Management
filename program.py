# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 12:38:17 2019

@author: abhijeet
"""

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Importing the dataset
dataset = pd.read_csv('Final_Dataset.csv')
dataset.drop('Unnamed: 0', axis=1, inplace=True)
X = dataset.loc[:, dataset.columns != 'Production'].values
y = dataset['Production'].values

regressor = RandomForestRegressor(random_state=0, n_jobs=-1,
                                  n_estimators=30,
                                  max_features='auto',
                                  max_depth=15,
                                  min_samples_split=6,
                                  bootstrap=True)

# Fitting Random Forest  to the Training set


regressor.fit(X, y)
print(regressor.score(X, y))

# Saving model

joblib.dump(regressor, 'model.sav')
