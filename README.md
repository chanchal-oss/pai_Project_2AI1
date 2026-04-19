# 📊 Linear Regression Models Collection



**Multiple Linear Regression Models using Different Datasets**

---

## 📖 Overview

This project demonstrates the implementation of **Linear Regression** on multiple real-world inspired datasets.
Each model predicts a continuous output based on different input features.

The goal of this project is to understand how the same machine learning algorithm can be applied across various domains.

---

## 🎯 Objectives

* Apply Linear Regression on different datasets
* Understand feature-target relationships
* Compare predictions across domains
* Build a reusable ML pipeline

---

## 🧠 Models Included

### 1. 📚 Student Marks Prediction

Predicts student marks based on:

* Study Hours
* Attendance
* Sleep Hours

---

### 2. 🏠 House Price Prediction

Predicts house price based on:

* Area
* Number of Bedrooms
* Age of House

---

### 3. 🚗 Car Mileage Prediction

Predicts mileage based on:

* Engine Size
* Weight
* Age

---

### 4. 🛍️ Sales Prediction

Predicts sales based on:

* Advertising Budget
* Discount Offered
* Store Visits

---

## ⚙️ Technologies Used

* Python 🐍
* Pandas
* Scikit-learn
* NumPy

---

## 🔄 Machine Learning Workflow

1. Data Collection (Synthetic Dataset)
2. Data Preprocessing
3. Feature Selection
4. Train-Test Split
5. Model Training using Linear Regression
6. Prediction
7. Evaluation (MSE, R² Score)

---

## 📂 Project Structure

```
📁 Linear-Regression-Project
│── student_marks.py
│── house_price.py
│── car_mileage.py
│── sales_prediction.py
│── README.md
```

---

## 📊 Sample Code Snippet

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
prediction = model.predict(X_test)
```

---

## 🚀 How to Run

1. Install dependencies:

```
pip install pandas scikit-learn
```

2. Run any file:

```
python student_marks.py
```

---

## 📈 Output

* Predicted values for each dataset
* Model evaluation metrics

---

## 🔥 Key Learning

* Linear Regression works for **continuous outputs**
* Same algorithm can solve multiple real-world problems
* Data plays the most important role in prediction accuracy

---

## 🚀 Future Improvements

* Add real datasets (CSV files)
* Visualization using Matplotlib
* Deploy using Flask
* Add GUI Interface

---



## ⭐ Conclusion

This project shows how Linear Regression can be applied across different domains with the same workflow, making it a powerful and versatile algorithm in Machine Learning.
