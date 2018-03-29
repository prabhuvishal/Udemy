# Simple linear regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
                
# train and test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)     

# Apply linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()           
regressor.fit(X_train,y_train)

# predict the output
y_pred = regressor.predict(X_test)

# Plot the training set
plt.scatter(X_train,y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary Vs. Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

# Plot the test set
plt.scatter(X_test,y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary Vs. Experience (Test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()


