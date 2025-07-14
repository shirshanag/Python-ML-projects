# 🛡️ Credit Card Fraud Detection System - LegitScan AI

A machine learning project to detect fraudulent credit card transactions using Logistic Regression. This system uses real transaction data and applies data preprocessing, under-sampling, and classification to predict whether a transaction is legitimate or fraudulent.

---

## 📂 Dataset

- **Source:** [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Classes:**
  - `0` → Legitimate Transaction
  - `1` → Fraudulent Transaction

---

## 📊 Problem Statement

The dataset is **highly imbalanced**, with only 492 fraudulent transactions out of ~285k total. This system uses **undersampling** to balance the dataset and applies **Logistic Regression** to classify transactions.

---

## ✅ Features

- Applied **Logistic Regression** to classify fraud vs legit transactions.
- Used **undersampling** technique for class balancing.
- Evaluated performance using **accuracy score**.
- Created a **prediction system** to classify new transactions.
- Easily extendable into a **Streamlit** or **Flask** app.

---

## 🧠 Tech Stack

- Python 🐍
- pandas, numpy
- scikit-learn
- Jupyter Notebook or Google Colab
