#importing Dependencies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
#Load the dataset
diabetes=pd.read_csv("/content/diabetes (1).csv")
#First five values
diabetes.head()
#No of rows and columns of the dataset
diabetes.shape
#Statistical meausres of the data
diabetes.describe()
#0--->Non Diabetic
#1--->Diabetic
diabetes["Outcome"].value_counts()



# 1. Separate features and labels
X = diabetes.drop(columns='Outcome', axis=1)
y = diabetes['Outcome']

# 2. Split data FIRST (This keeps the test data hidden)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# 3. Standardize data SECOND
scaler = StandardScaler()

# Fit the scaler ONLY on training data
X_train = scaler.fit_transform(X_train)

# Transform the test data using the training mean/std (DO NOT FIT ON TEST)
X_test = scaler.transform(X_test)

# 4. Model Training (The rest stays the same)
model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)
#Model prediction for train data
x_train_predict=model.predict(X_train)
x_train_accuracy=accuracy_score(x_train_predict,y_train)
print("Accuracy on train data:",x_train_accuracy)
#Model prediction for test data
x_test_predict=model.predict(X_test)
x_test_accuracy=accuracy_score(x_test_predict,y_test)
print("Accuracy on test data:",x_test_accuracy)
#Build Seaborn
cm = confusion_matrix(y_test, x_test_predict)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
f1=f1_score(y_test,x_test_predict)
print("F1 Score:",f1)
#Building a Predictive system
# Predictive System - Clean version
input_data = (5, 109, 75, 26, 0, 36, 0.546, 60)

# Create DataFrame with column names
feature_names = diabetes.drop(columns='Outcome').columns
input_df = pd.DataFrame([input_data], columns=feature_names)

# Standardize input
std_data = scaler.transform(input_df)

# Predict
prediction = model.predict(std_data)

# Output
print("Prediction:", prediction[0])
if prediction[0] == 0:
    print("The person is not diabetic")
else:
    print("The person is diabetic")
