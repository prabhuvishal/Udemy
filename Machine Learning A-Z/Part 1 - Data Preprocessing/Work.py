# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
 
# Add missing values                
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy ='median', axis = 0)                
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

# Convert categorical variables to integers

# while encoding to multiple columns, use labelencoder first and OneHotEncoder next
# else use LabelEncoder alone
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)


# train and test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_Test = train_test_split(X,y,test_size = 0.2, random_state = 42)

# feature scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train[:,3:5] = sc_X.fit_transform(X_train[:,3:5])
X_test[:,3:5] = sc_X.transform(X_test[:,3:5])