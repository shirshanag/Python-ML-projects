#Import dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#Load the dataset
df=pd.read_csv("/content/creditcard.csv")
#First five values
df.head()
#0----> Legitemate Trasnsaction
#1----> Fraudulent Transaction
#Dataset is highly unbalanced
df["Class"].value_counts()
#Seperating for analysis
legit=df[df.Class==0]
fraud=df[df.Class==1]

#Statistics of the amount
legit.Amount.describe()
fraud.Amount.describe()
#Compare the values for both transactions
df.groupby('Class').mean()
#Under-Sampling
#Build a dataset containing similar no of legit and fraudulent
legit_sample=legit.sample(n=492)
print(legit_sample.shape)
#Concatenating Dataframe
new_df=pd.concat([legit_sample,fraud],axis=0)
new_df.head()
#Seperate features and labels
X=new_df.drop(columns='Class',axis=1)
y=new_df['Class']
#Train test split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=2)
#Model--->Logistic Regression Model
model=LogisticRegression()
#Fit Model
model.fit(X_train,y_train)
#Predict based on train data
x_train_predict=model.predict(X_train)
x_train_accuracy=accuracy_score(x_train_predict,y_train)
print("Accuracy on train data out of 1:",x_train_accuracy)
#Predict based on test data
x_test_predict=model.predict(X_test)
x_test_accuracy=accuracy_score(x_test_predict,y_test)
print("Accuracy on test data out of 1:",x_test_accuracy)
