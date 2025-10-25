import streamlit as st
import pandas as pd
import uuid
from datetime import datetime
import os

# ---------- Utility Functions ----------
DATA_FILE = "harder_synthetic_tasks.csv"

def load_data():
    expected_columns = ["id", "task_name", "description", "assigned_to", "priority",
                        "status", "start_date", "due_date", "notes"]

    if os.path.exists(DATA_FILE):
        # Read the CSV
        df = pd.read_csv(DATA_FILE)

        # Clean and rename columns
        df.columns = df.columns.str.strip().str.lower()

        rename_map = {
            "task id": "id",
            "task name": "task_name",
            "assigned to": "assigned_to",
            "start date": "start_date",
            "due date": "due_date"
        }

        df.rename(columns=rename_map, inplace=True)

        # Add missing expected columns
        for col in expected_columns:
            if col not in df.columns:
                df[col] = None

        # Convert date columns
        for date_col in ["start_date", "due_date"]:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)

        # Fill empty text fields with empty strings (so Streamlit displays them)
        df = df.fillna("")

        return df[expected_columns]
    else:
        # Return an empty DataFrame with correct columns
        return pd.DataFrame(columns=expected_columns)

    # if os.path.exists(DATA_FILE):
    #     return pd.read_csv(DATA_FILE, parse_dates=["start_date", "due_date"])
    # else:
    #     return pd.DataFrame(columns=["id", "task_name", "assigned_to", "priority", "status", "start_date", "due_date", "notes"])

def save_data(df):
    df.to_csv(DATA_FILE, index=False)

def add_task(new_task):
    df = load_data()
    df = pd.concat([df, pd.DataFrame([new_task])], ignore_index=True)
    save_data(df)

# ---------- AI Priority Classifier ----------
def classify_priority(text):
    text = text.lower()
    urgent_keywords = ["deadline", "submit", "urgent", "immediately", "asap", "critical"]
    medium_keywords = ["meeting", "prepare", "review", "follow up", "remind"]
    if any(word in text for word in urgent_keywords):
        return "high"
    elif any(word in text for word in medium_keywords):
        return "medium"
    return "low"

# ---------- Streamlit UI ----------
st.set_page_config(page_title="AI Task Manager", layout="wide")
st.title("🧠 AI-Powered Task Manager")

# ---------- Sidebar: Add Task ----------
with st.sidebar:
    st.header("➕ Add New Task")

    st.markdown("### 🧠 AI Priority Suggestion")
    ai_task_desc = st.text_area("Describe your task here")

    if st.button("⚡ Predict Priority"):
        if ai_task_desc.strip():
            predicted_priority = classify_priority(ai_task_desc)
            st.success(f"Predicted Priority: **{predicted_priority.upper()}**")
            st.session_state["ai_task_name"] = ai_task_desc
            st.session_state["ai_predicted_priority"] = predicted_priority
        else:
            st.warning("Please enter a task description")

    st.markdown("---")
    st.markdown("### 📌 Manual or Suggested Task Entry")

    task_name = st.text_input("Task Name", value=st.session_state.get("ai_task_name", ""))
    assigned_to = st.text_input("Assigned To")
    priority = st.selectbox("Priority", ["low", "medium", "high"],
                            index=["low", "medium", "high"].index(st.session_state.get("ai_predicted_priority", "medium")))
    status = st.selectbox("Status", ["pending", "in progress", "completed"])
    start_date = st.date_input("Start Date", datetime.today())
    due_date = st.date_input("Due Date", datetime.today())
    notes = st.text_area("Notes")

    if st.button("Add Task"):
        if task_name:
            new_task = {
                "id": str(uuid.uuid4()),
                "task_name": task_name,
                "assigned_to": assigned_to,
                "priority": priority,
                "status": status,
                "start_date": pd.to_datetime(start_date),
                "due_date": pd.to_datetime(due_date),
                "notes": notes
            }
            add_task(new_task)
            st.success("Task added successfully!")
            st.session_state["ai_task_name"] = ""
            st.session_state["ai_predicted_priority"] = "medium"
            st.rerun()
        else:
            st.warning("Task name is required")

# ---------- Main Area: Task Table ----------
st.subheader("📋 Your Tasks")
df = load_data()

if df.empty:
    st.info("No tasks available. Add a new task from the sidebar.")
else:
    status_filter = st.multiselect("Filter by Status", options=df["status"].unique(), default=df["status"].unique())
    priority_filter = st.multiselect("Filter by Priority", options=df["priority"].unique(), default=df["priority"].unique())
    assigned_to_filter = st.multiselect("Filter by Assignee", options=df["assigned_to"].unique(), default=df["assigned_to"].unique())

    filtered_df = df[
        (df["status"].isin(status_filter)) &
        (df["priority"].isin(priority_filter)) &
        (df["assigned_to"].isin(assigned_to_filter))
    ]

    st.dataframe(filtered_df.sort_values("due_date"))
