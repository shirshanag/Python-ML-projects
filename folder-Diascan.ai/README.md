# Diascan.ai 🩺 - Diabetes Prediction System for Females using RFC

**Diascan.ai** is a machine learning project focused on predicting diabetes in female patients using data from the PIMA Indian Diabetes dataset. The model is built with a Random Forest  and optimized for medical relevance, using proper feature scaling and metric-based evaluation.

---

## 🔍 Problem Statement

Early detection of diabetes is crucial for improving health outcomes. This project targets **female-specific prediction** by leveraging features such as pregnancies, glucose levels, BMI, and more to predict the presence of diabetes (binary classification).

---

## 🧠 Technologies Used

- **Language**: Python  
- **Libraries**: `scikit-learn`, `pandas`, `matplotlib`, `seaborn`  
- **Model**: Random Forest Classifier  
- **Evaluation**: F1-score, Confusion Matrix, Accuracy, Recall  
- **Scaling**: StandardScaler  
- **Dataset**: Kaggle

## 📊 Features Used

- Number of pregnancies  
- Glucose concentration  
- Blood pressure  
- Skin thickness  
- Insulin  
- BMI  
- Diabetes pedigree function  
- Age
## ⚙️ How It Works

1. Data Cleaning (handling zero values in BMI, glucose, etc.)
2. Data Scaling using `StandardScaler`
3. Train-Test Split (80/20)
4. Random Forest Model Training 
5. Evaluation using F1-score and Confusion Matrix
6. ## 🧪 Sample Code Snippet

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1. Feature Selection (Dropping noisy features based on importance)
# We found SkinThickness and Insulin were adding noise/overfitting
X = diabetes.drop(columns=['Outcome', 'SkinThickness', 'Insulin', 'BloodPressure'], axis=1)
y = diabetes['Outcome']

# 2. Split data FIRST 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# 3. Random Forest Model with Regularization
# We use max_depth and min_samples_leaf to prevent the 1.0 Training Accuracy issue
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=4, 
    min_samples_leaf=5,
    class_weight='balanced',  # Helps improve the F1-Score for the minority class
    random_state=2
)

model.fit(X_train, y_train)

# 4. Evaluation
y_pred = model.predict(X_test)
print(f"Training Accuracy: {model.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {model.score(X_test, y_test):.4f}")
print(classification_report(y_test, y_pred))
