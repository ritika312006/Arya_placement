import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Arya College Placements Dashboard", layout="centered")

st.title("🎓 Arya College Placement Dashboard (2010 - 2025)")
st.markdown("Welcome! This dashboard shows placement trends of Arya College and predicts placements for 2026 using Linear Regression.")

# Load CSV file
file_path = r"C:\Users\Hi\Arya_placement\college_placements_2010_2025_with_company (3).csv"
df = pd.read_csv(file_path)

# Auto-detect column name for placed students
possible_columns = ['Placed_Students', 'Placed Students', 'placed_students', 'Placed', 'TotalPlaced']
placed_col = None

for col in df.columns:
    if col.strip() in possible_columns:
        placed_col = col
        break

# Validate required columns
if placed_col is None or 'Year' not in df.columns:
    st.error("❌ Required columns not found. Make sure your CSV includes 'Year' and one of these: " + ", ".join(possible_columns))
    st.stop()

# Display CSV data
st.subheader("📄 Full Placement Data (2010 - 2025)")
st.dataframe(df)

# Summary KPIs
total_placed = df[placed_col].sum()
avg_per_year = df[placed_col].mean()

st.subheader("📊 Placement Summary")
st.metric("Total Placed Students", f"{total_placed}")
st.metric("Average Per Year", f"{int(avg_per_year)}")

# Bar chart: Year-wise placement trend
st.subheader("📈 Year-wise Placement Trend")
fig, ax = plt.subplots()
ax.bar(df['Year'], df[placed_col], color='royalblue')
ax.set_xlabel("Year")
ax.set_ylabel("Number of Students Placed")
ax.set_title("Placements per Year")
st.pyplot(fig)

# Prediction using Linear Regression
X = df[['Year']]
y = df[[placed_col]]

model = LinearRegression()
model.fit(X, y)

next_year = 2026
predicted_2026 = int(model.predict([[next_year]])[0][0])

# Display predicted data in a separate table
st.subheader("📅 Prediction for 2026")
pred_df = pd.DataFrame({
    'Year': [2026],
    'Predicted_Placed_Students': [predicted_2026]
})
st.table(pred_df)

# Bar chart with prediction included
st.subheader("📊 Placements Including 2026 Forecast")
all_years = df['Year'].tolist() + [2026]
all_placements = df[placed_col].tolist() + [predicted_2026]

fig2, ax2 = plt.subplots()
colors = ['royalblue'] * len(df) + ['mediumseagreen']
bars = ax2.bar(all_years, all_placements, color=colors)
ax2.set_xlabel("Year")
ax2.set_ylabel("Placements")
ax2.set_title("Placement Trend (2010-2026 with Prediction)")
ax2.bar_label(bars, padding=3)
st.pyplot(fig2)

st.success("✅ Dashboard and 2026 prediction loaded successfully!")
st.caption("Made by Ritika | Arya College")
