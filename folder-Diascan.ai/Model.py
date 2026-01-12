# Importing Dependencies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
diabetes = pd.read_csv("/content/diabetes (1).csv")

# Separate features and labels
X = diabetes.drop(columns='Outcome', axis=1)
y = diabetes['Outcome']
# 2. Split data FIRST (This keeps the test data hidden)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 3. Standardize data SECOND
scaler = StandardScaler()

# Fit the scaler ONLY on training data
X_train = scaler.fit_transform(X_train)

# Transform the test data using the training mean/std (DO NOT FIT ON TEST)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,              # Slightly shallower to prevent memorizing noise
    min_samples_leaf=5,       # Each 'leaf' must have at least 5 patients
    class_weight='balanced',      # Forces trees to be different from each other
    random_state=2
)
model.fit(X_train, y_train)

# Predict on train and test
x_train_predict = model.predict(X_train)
x_test_predict = model.predict(X_test)

# Accuracy
print("Accuracy on train data:", accuracy_score(y_train, x_train_predict))
print("Accuracy on test data:", accuracy_score(y_test, x_test_predict))

# F1 Score (on test data)
f1 = f1_score(y_test, x_test_predict)
print("F1 Score (Test):", f1)

# Confusion Matrix
cm = confusion_matrix(y_test, x_test_predict)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Diabetic', 'Diabetic'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.show()

# Predictive System
input_data = (5, 109, 75, 26, 0, 36, 0.546, 60)
feature_names = diabetes.drop(columns='Outcome').columns
input_df = pd.DataFrame([input_data], columns=feature_names)
std_data = scaler.transform(input_df)
prediction = model.predict(std_data)

print("\nPrediction:", prediction[0])
if prediction[0] == 0:
    print("The person is not diabetic")
else:
    print("The person is diabetic")
