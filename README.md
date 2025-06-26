# Ai-powered-task-manager
# 🧠 AI-Based Task Management System

This project is a Machine Learning-powered system designed to analyze and predict task outcomes using features like task priority, status, duration, and description. Built entirely in a Jupyter Notebook, the system showcases a practical implementation of AI in task automation and performance prediction.

---

## 📌 Project Objectives

- Predict outcomes of tasks using historical task data
- Apply text processing on task descriptions
- Train a Logistic Regression model to classify tasks
- Evaluate the model using key metrics and a confusion matrix
- Export evaluation results for dashboard integration

---

## 🗂️ Features

- 🔍 **Data Cleaning & Preprocessing**  
  Handles null values, encodes categorical data, and normalizes task durations.

- ⚖️ **Workload Balancing**  
  Redistributes tasks fairly among users.

- ✨ **Text Processing**  
  Cleans task descriptions by removing punctuation, stopwords, and applying lemmatization.

- 📈 **Model Training & Evaluation**  
  - Logistic Regression using Scikit-learn  
  - Train/Test Split  
  - Classification report with metrics (Precision, Recall, F1-Score)  
  - Confusion Matrix heatmap  

- 💾 **Exported Results**  
  Saves model evaluation as:
  - `metrics.json`
  - `confusion_matrix.csv`
  - `best_params.json`

---

## 🧰 Technologies Used

- Python 3.x  
- Jupyter Notebook  
- Pandas  
- Scikit-learn  
- NLTK  
- Matplotlib / Seaborn

---


