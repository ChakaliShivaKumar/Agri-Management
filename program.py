# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Importing the dataset
dataset = pd.read_csv('Final_Dataset.csv')
X = dataset.loc[:, dataset.columns != 'Production'].values
y = dataset['Production'].values

regressor = RandomForestRegressor(random_state=0, n_jobs=-1,
                                  n_estimators=30,
                                  max_depth=15,
                                  min_samples_split=6,
                                  bootstrap=True)

# Fitting Random Forest  to the Training set


regressor.fit(X, y)
print(regressor.score(X, y))

# Saving model

joblib.dump(regressor, 'model.sav')
