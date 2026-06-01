# 🏏 Cricket Score Prediction using Machine Learning

## 📌 Overview

Cricket Score Prediction is a Machine Learning project that predicts the final score of a cricket team during an innings using historical IPL match data. The project applies multiple regression algorithms and compares their performance to identify the most accurate model for score forecasting.

The system leverages match-related features such as batting team, bowling team, venue, current score, wickets, overs, batsman, and bowler information to estimate the final innings total.

---

## 🎯 Objectives

* Predict the final score of a cricket innings using machine learning.
* Compare the performance of multiple regression models.
* Analyze the impact of different cricket match features on prediction accuracy.
* Build a reliable sports analytics solution for score forecasting.

---

## 📊 Dataset

The project uses an IPL (Indian Premier League) ball-by-ball dataset containing:

* Match details
* Batting and bowling teams
* Venue information
* Batsman and bowler statistics
* Current runs and wickets
* Overs completed
* Final innings score (target variable)

### Key Features

* `bat_team`
* `bowl_team`
* `venue`
* `batsman`
* `bowler`
* `runs`
* `wickets`
* `overs`
* `runs_last_5`
* `wickets_last_5`

### Target Variable

* `total` → Final innings score

---

## ⚙️ Data Preprocessing

The following preprocessing steps were performed:

* Removal of unnecessary columns.
* Filtering innings with less than 5 overs.
* Handling categorical variables using One-Hot Encoding.
* Grouping less frequent venues into an "Other_Venue" category.
* Feature selection and transformation.
* Dataset splitting into Training, Validation, and Testing sets.

---

## 🤖 Machine Learning Models Used

### 1. Linear Regression

A baseline regression model that assumes a linear relationship between features and target score.

### 2. Ridge Regression

Linear regression with L2 regularization to reduce overfitting.

### 3. Lasso Regression

Linear regression with L1 regularization that can eliminate less important features.

### 4. Elastic Net Regression

Combination of Ridge and Lasso regularization.

### 5. Random Forest Regression

An ensemble learning model that combines multiple decision trees and captures complex nonlinear relationships.

---

## 📈 Evaluation Metrics

The models were evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* R² Score (Coefficient of Determination)

---

## 🏆 Results

### Validation Set Performance

| Model                        | MAE      | MSE       | R² Score   |
| ---------------------------- | -------- | --------- | ---------- |
| Linear Regression            | 11.93    | 256.19    | 0.7005     |
| Ridge Regression             | 12.00    | 257.60    | 0.6989     |
| Lasso Regression             | 19.83    | 642.59    | 0.2488     |
| Elastic Net Regression       | 21.25    | 740.61    | 0.1342     |
| **Random Forest Regression** | **4.05** | **44.04** | **0.9485** |

### Test Set Performance

| Model                        | MAE      | MSE       | R² Score   |
| ---------------------------- | -------- | --------- | ---------- |
| Linear Regression            | 12.03    | 255.22    | 0.6985     |
| Ridge Regression             | 12.09    | 257.23    | 0.6961     |
| Lasso Regression             | 19.87    | 645.60    | 0.2373     |
| Elastic Net Regression       | 21.22    | 737.24    | 0.1290     |
| **Random Forest Regression** | **4.10** | **43.55** | **0.9486** |

### Best Model

✅ **Random Forest Regression** achieved the highest prediction accuracy with an R² score of **0.9486**, significantly outperforming the other regression models.

---

## 📉 Visualizations

The project includes:

* Actual vs Predicted Score plots
* R² Score comparison charts
* Model performance analysis

These visualizations help understand prediction quality and compare model effectiveness.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Machine Learning
* Regression Analysis

---

## 🚀 Project Workflow

1. Data Collection
2. Data Cleaning
3. Feature Engineering
4. Data Encoding
5. Train-Test Split
6. Model Training
7. Model Evaluation
8. Performance Comparison
9. Visualization & Analysis

---

## 📂 Project Structure

```text
Cricket-Score-Prediction/
│
├── ipl_data.csv
├── Sayan_Ml_Code.py
├── Linear.png
├── Ridge.png
├── Lasso.png
├── Net.png
├── RndFor.png
├── R2_val.png
├── R2_test.png
├── README.md
└── Project_Report.pdf
```

---

## 🔮 Future Improvements

* Incorporate live match data streams.
* Use Deep Learning models such as LSTM and GRU.
* Add player form and historical performance metrics.
* Develop a real-time score prediction dashboard.
* Deploy the model as a web application using Flask or Streamlit.

---

## 📚 Conclusion

This project demonstrates the application of Machine Learning in sports analytics by predicting cricket innings scores using IPL match data. Among all evaluated models, Random Forest Regression delivered the best performance, proving that ensemble learning techniques are highly effective for capturing complex cricket scoring patterns.
