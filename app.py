import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("StudentsPerformance.csv")

data = data.rename(columns={
    "math score": "math",
    "reading score": "reading",
    "writing score": "writing"
})

data["english"] = (data["reading"] + data["writing"]) / 2

X = data[["reading", "writing"]]
y = data["english"]
model = LinearRegression()
model.fit(X, y)
# UI
st.title("Student Performance Predictor")
st.write("This app predicts a student's english score based on reading and writing performance.")

st.markdown("### Enter Student Details")

reading = st.slider("Reading Score", 0, 100)
writing = st.slider("Writing Score", 0, 100)

if st.button("Predict"):
    input_data = [[reading, writing]]
    prediction = model.predict(input_data)[0]

    st.success(f"Predicted english Score: {prediction:.2f}")

    # Graph
    st.subheader("Model Feature Importance (Static)")

features = ["Reading", "Writing"]
importance = model.coef_

fig, ax = plt.subplots()
ax.bar(features, importance)
st.pyplot(fig)

st.subheader("Feature Contribution (Dynamic)")

values = [reading, writing]
contribution = [values[i] * importance[i] for i in range(len(values))]

fig, ax = plt.subplots()
ax.bar(features, contribution)
st.pyplot(fig)