import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.metrics import mean_absolute_error, r2_score

from train_model import train_and_save_model, DATA_PATH, MODEL_PATH, FEATURES

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

st.title("🎓 Student Performance Prediction & Analysis")
st.write(
    "An educational machine-learning demo that estimates student marks from "
    "study habits and academic indicators. The included dataset is synthetic."
)

if not os.path.exists(MODEL_PATH):
    with st.spinner("Training the prediction model for the first time..."):
        train_and_save_model()

model = joblib.load(MODEL_PATH)
data = pd.read_csv(DATA_PATH)

st.sidebar.header("Enter Student Details")
study_hours = st.sidebar.slider("Daily study hours", 0.5, 8.0, 3.5, 0.5)
attendance = st.sidebar.slider("Attendance (%)", 45, 100, 80)
previous_score = st.sidebar.slider("Previous exam score", 25, 100, 65)
assignments = st.sidebar.slider("Assignments completed (out of 10)", 0, 10, 7)
sleep_hours = st.sidebar.slider("Average sleep (hours)", 4.0, 9.5, 7.0, 0.5)
extracurricular = st.sidebar.selectbox("Participates in extracurricular activities?", ["No", "Yes"])

input_df = pd.DataFrame([{
    "study_hours": study_hours,
    "attendance": attendance,
    "previous_score": previous_score,
    "assignments_completed": assignments,
    "sleep_hours": sleep_hours,
    "extracurricular": 1 if extracurricular == "Yes" else 0
}])

prediction = float(model.predict(input_df[FEATURES])[0])
prediction = float(np.clip(prediction, 0, 100))

if prediction >= 75:
    category, advice = "Good performance range", "Keep maintaining consistent study habits and attendance."
elif prediction >= 50:
    category, advice = "Average performance range", "A structured revision plan and regular practice may help."
else:
    category, advice = "Needs academic support", "Consider additional guidance, practice, and a realistic study schedule."

c1, c2, c3 = st.columns(3)
c1.metric("Predicted marks", f"{prediction:.1f} / 100")
c2.metric("Performance category", category)
c3.metric("Attendance", f"{attendance}%")
st.info(f"**Suggestion:** {advice}")

st.subheader("Your entered academic profile")
st.dataframe(input_df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Dataset Analysis")
left, right = st.columns(2)

with left:
    fig = px.scatter(
        data, x="study_hours", y="final_marks", color="attendance",
        title="Study Hours vs Final Marks",
        labels={"study_hours": "Daily study hours", "final_marks": "Final marks", "attendance": "Attendance (%)"}
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.histogram(
        data, x="final_marks", nbins=20,
        title="Distribution of Final Marks",
        labels={"final_marks": "Final marks"}
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Average Marks by Attendance Group")
attendance_data = data.copy()
attendance_data["attendance_group"] = pd.cut(
    attendance_data["attendance"],
    bins=[0, 60, 75, 90, 101],
    labels=["Below 60%", "60–75%", "75–90%", "Above 90%"]
)
grouped = attendance_data.groupby("attendance_group", observed=False)["final_marks"].mean().reset_index()
fig = px.bar(grouped, x="attendance_group", y="final_marks", title="Average Marks by Attendance")
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Important: This is a learning project using synthetic sample data. "
    "Predictions are estimates, not official grades or decisions about a student."
)
