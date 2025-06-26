# dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import json

st.set_page_config(page_title="Task Dashboard", layout="wide")

# Load data
st.title("Task Management Dashboard")
uploaded_file = st.file_uploader("Upload processed CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.header("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Info")
    st.write("Shape of Dataset:", df.shape)
    st.write("Missing Values:")
    st.write(df.isnull().sum())

    # Distribution of Task Priority
    st.subheader("Task Priority Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='priority', order=df['priority'].value_counts().index, palette='Reds', ax=ax1)
    ax1.set_title("Priority Distribution")
    st.pyplot(fig1)

    # Task Status Distribution
    st.subheader("Task Status Distribution")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df, x='status', order=df['status'].value_counts().index, palette='Blues', ax=ax2)
    ax2.set_title("Status Distribution")
    st.pyplot(fig2)

    # Task Duration Histogram
    st.subheader("Task Duration (Days)")
    fig3, ax3 = plt.subplots()
    sns.histplot(df['task_duration_days'], bins=10, kde=True, color='green', ax=ax3)
    ax3.set_title("Task Duration")
    ax3.set_xlabel("Days")
    st.pyplot(fig3)

    # Tasks Assigned Per User
    st.subheader("Top 10 Users by Task Assignment")
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    top_users = df['assigned to'].value_counts().head(10)
    top_users.plot(kind='bar', color='orange', ax=ax4)
    ax4.set_title('Top 10 Users by Tasks Assigned')
    ax4.set_xlabel('User')
    ax4.set_ylabel('Task Count')
    plt.xticks(rotation=45)
    st.pyplot(fig4)

    st.subheader("Logistic Regression Model Evaluation")

    # Upload and load metrics
    metrics_file = st.file_uploader("Upload model metrics (metrics.json)", type=["json"])
    cm_file = st.file_uploader("Upload confusion matrix (confusion_matrix.csv)", type=["csv"])
    params_file = st.file_uploader("Upload best parameters (best_params.json)", type=["json"])

    if metrics_file and cm_file and params_file:
        # Load metrics
        metrics = json.load(metrics_file)
        accuracy = metrics["weighted avg"]["precision"]
        precision = metrics["weighted avg"]["precision"]
        recall = metrics["weighted avg"]["recall"]
        f1 = metrics["weighted avg"]["f1-score"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy*100:.2f}%")
        col2.metric("Precision", f"{precision*100:.2f}%")
        col3.metric("Recall", f"{recall*100:.2f}%")
        col4.metric("F1 Score", f"{f1*100:.2f}%")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = pd.read_csv(cm_file).values
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        # Best Params
        st.subheader("Best Hyperparameters from GridSearchCV")
        best_params = json.load(params_file)
        st.json(best_params)

    else:
        st.info("Upload all 3 files (metrics.json, confusion_matrix.csv, best_params.json) to view model evaluation.")