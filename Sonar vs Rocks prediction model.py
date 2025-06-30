#Import Dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#Data Collection and Data Preprocessing
sonar_data=pd.read_csv('Copy of sonar data.csv',header=None)
#First five rows
sonar_data.head()
#No of rows and columns
sonar_data.shape
#Statistical measures
sonar_data.describe()
#No of Rock and mine
sonar_data[60].value_counts()
#Groupby
sonar_data.groupby(60).mean()
#Seperating data and Labels
#labels-> M or R Data-> Numeric value
X=sonar_data.drop(columns=60,axis=1)
y=sonar_data[60]
#Split Dataset into train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,stratify=y,random_state=1)
#Model training---->Logistic Regression
model=LogisticRegression()
#Fiting the model
model.fit(X_train,y_train)
#Model Predict for train data
X_train_prediction=model.predict(X_train)
#Accuracy on test data
training_data_accuracy=accuracy_score(X_train_prediction,y_train)
print("Accuracy:",training_data_accuracy)
#Accuracy on train data
X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,y_test)
print("Accuracy:",test_data_accuracy)
