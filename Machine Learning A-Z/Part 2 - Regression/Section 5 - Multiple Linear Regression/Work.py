# Mulitple linear regresion

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Azure_Intent_Dataset.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

"""
# Convert categorical variables to integers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()


#Remove first column (Dummy trap)
X = X[:,1:]
"""

# train and test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, Y_Test = train_test_split(X,y,test_size = 0.2, random_state = 42)

# Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predict
y_pred = regressor.predict(X_test)

# Backward elimination logic
import statsmodels.formula.api as sm
X = np.append(arr=np.ones((len(X),1)).astype(int),values=X,axis =1 )

X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()    
regressor_OLS.summary()

