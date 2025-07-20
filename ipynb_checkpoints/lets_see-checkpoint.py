import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Load data
@st.cache_data
def load_data():
    # ✅ Make sure this file exists in the same folder as lets_see.py
    return pd.read_csv("college_placements_2010_2025_with_company.csv")

df = load_data()

# App title
st.title("Arya College Placement Analysis (2010–2025)")
st.markdown("📈 This app visualizes placement trends and predicts 2026 package using regression.")

# Show data
if st.checkbox("Show Raw Placement Data"):
    st.dataframe(df)

# Year-wise average package plot
st.subheader("📊 Year-wise Average Package Trend")
fig, ax = plt.subplots()
ax.plot(df['Year'], df['Average Package (LPA)'], marker='o', color='blue', label='Avg Package')
ax.set_xlabel("Year")
ax.set_ylabel("Average Package (LPA)")
ax.set_title("Placement Trends from 2010 to 2025")
ax.grid(True)
st.pyplot(fig)

# Show company data
st.subheader("🏢 Company-Wise Placement Summary")
company_group = df.groupby("Company")["No. of Students Placed"].sum().sort_values(ascending=False)
st.bar_chart(company_group)

# Predict 2026 average package using Linear Regression
st.subheader("🔮 Predicted Average Package for 2026")

# Prepare regression model
X = df[['Year']]
y = df['Average Package (LPA)']

model = LinearRegression()
model.fit(X, y)

predicted_2026 = model.predict([[2026]])
st.success(f"📌 Predicted Avg Package for 2026: **{predicted_2026[0]:.2f} LPA**")

# Footer
st.markdown("---")
st.markdown("🔗 [Arya College Placements Website](https://www.aryacollege.in/placements)")
