# 🧠 Logistic Regression Classifier

This project is part of the AI & ML , focused on building a **binary classification model** using **Logistic Regression**. The dataset used is the **Breast Cancer Wisconsin Dataset**, and the goal is to classify whether a tumor is benign or malignant.

---

## 📌 Objective

- Build a logistic regression model for binary classification.
- Learn how to evaluate classification models using various metrics.
- Understand the sigmoid function and threshold tuning.

---

## 🧰 Tools Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## 📁 Dataset

**Breast Cancer Wisconsin Dataset**  
[Download from Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

---

## 📊 Workflow

### 1. **Data Loading**
- Load dataset into a Pandas DataFrame.
- Inspect for missing values or irrelevant columns.

### 2. **Preprocessing**
- Dropped non-essential columns (like `Id`).
- Mapped diagnosis labels: `M` → 1 (Malignant), `B` → 0 (Benign).
- Imputed missing values using `SimpleImputer`.
- Standardized features using `StandardScaler`.

### 3. **Model Training**
- Split the dataset into training and testing sets (80/20).
- Trained a logistic regression model using `sklearn.linear_model.LogisticRegression`.

### 4. **Model Evaluation**
- Evaluated the model using:
  - **Confusion Matrix**
  - **Precision, Recall, F1-Score**
  - **ROC-AUC Curve**
- Plotted confusion matrix heatmap and ROC curve.

### 5. **Threshold Tuning (Optional)**
- Changed the decision threshold to analyze its impact on precision/recall.

---

## 📈 Results

- Achieved high precision and recall on test data.
- ROC-AUC score showed excellent class separability.

---

## 📚 Concepts Covered

- Logistic Regression
- Sigmoid Function
- Binary Classification
- Confusion Matrix
- Precision, Recall, F1-score
- ROC Curve and AUC
- Threshold Tuning

---

## 🚀 How to Run

1. Clone the repository.
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
