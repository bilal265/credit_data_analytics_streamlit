import streamlit as st
import pandas as pd 
import numpy as np



# Part 1: Load the Data
@st.cache_data
def load_data():
    data = pd.read_csv('german_credit_data_processed.csv')
    return data

# Load data
df = load_data()

# Show success message
st.success("Data loaded successfully!")

# Optional: Display the data (you can remove this if not needed)
st.dataframe(df)

# Part 2: Show a Preview
st.header("Preview of the Dataset")

# Multiselect widget to choose columns
columns = st.multiselect(
    "Select columns to display:",
    options=df.columns.tolist(),
    default=df.columns.tolist()  # By default, show all columns
)

# Show first few rows of the selected columns
st.dataframe(df[columns].head())

# ✅ Part 3: Add Bar Chart for Value Counts
st.header("Bar Chart for Value Counts")

# Select one column
selected_column = st.selectbox(
    "Select a column to visualize:",
    options=df.columns.tolist()
)

# Check if the column is categorical or has < 20 unique values
if df[selected_column].dtype == 'object' or df[selected_column].nunique() < 20:
    # Show bar chart
    value_counts = df[selected_column].value_counts()
    st.bar_chart(value_counts)
else:
    # Show warning
    st.warning("Selected column has too many unique values or is not categorical. Please select another column.")

# Part 4: Add Line Chart for Numerical Columns
st.header("Line Chart for Numerical Columns")

# Get only numeric columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Multiselect to choose numeric columns
selected_numeric_columns = st.multiselect(
    "Select numeric columns to plot line chart:",
    options=numeric_columns
)

# Show line chart if any columns selected
if selected_numeric_columns:
    st.line_chart(df[selected_numeric_columns])
else:
    st.info("Please select at least one numeric column to display the line chart.")

# Part 5: Show a Correlation Table
st.header("Correlation Matrix")

# Calculate and show correlation matrix
correlation_matrix = df[numeric_columns].corr()
st.dataframe(correlation_matrix)

# Collapsible Statistics Section
st.header("Dataset Statistics")

with st.expander("Show Summary Statistics"):
    st.dataframe(df.describe())
