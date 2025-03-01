from sklearn.datasets import load_diabetes
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = load_diabetes()
print(type(df))
print(df)
print(pd.DataFrame(df.data))

dataset = pd.DataFrame(df.data)
dataset.columns = df.feature_names  # column as feature_names

# To display some first few rows of dataset
print(dataset.head())

dataset["patients"] = df.target
print(dataset.head())

# Dividing the independent and dependent features
x = dataset.iloc[: , :-1] # independent features that is all columns except patients
y = dataset.iloc[:,-1] # dependent features , here patients
print(x.head())
print(y.head())

# performing linear regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
lin_reg = LinearRegression()
# doing cross validation(cv) - In cross validation , we train model by giving some number of test data and training data continuously.
mse = cross_val_score(lin_reg , x , y , scoring = "neg_mean_squared_error" , cv = 5)
# mse is mean squared error . Here cv = 5 means that 5 times cross validation is perforomed
print(mse)  # These are 5 test data that are given to model
mean_mse = np.mean(mse) # using numpy to calculated mean of mse
print(mean_mse)