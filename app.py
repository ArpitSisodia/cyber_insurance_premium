import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# Generate synthetic data
np.random.seed(42)
n_samples = 500

# Company attributes
company_size = np.random.randint(50, 10000, n_samples)  # Number of employees
industry_risk = np.random.choice([1, 2, 3, 4, 5], n_samples)  # Risk level (1: Low, 5: High)
past_incidents = np.random.randint(0, 10, n_samples)  # Number of past cyber incidents
security_measures = np.random.randint(1, 6, n_samples)  # Security rating (1: Poor, 5: Excellent)
compliance = np.random.choice([0, 1], n_samples)  # 1 if compliant, 0 otherwise

# Define premium based on attributes
base_premium = 5000
premium = (
    base_premium + (company_size * 0.5) + (industry_risk * 2000) + (past_incidents * 1500)
    - (security_measures * 1000) - (compliance * 3000) + np.random.normal(0, 2000, n_samples)
)

# Ensure minimum premium
premium = np.clip(premium, 2000, None)

# Create DataFrame
data = pd.DataFrame({
    "Company Size": company_size,
    "Industry Risk": industry_risk,
    "Past Incidents": past_incidents,
    "Security Measures": security_measures,
    "Compliance": compliance,
    "Premium": premium
})

# Fit a simple regression model to understand impact of variables
X = data[["Company Size", "Industry Risk", "Past Incidents", "Security Measures", "Compliance"]]
y = data["Premium"]

model = LinearRegression()
model.fit(X, y)

coefficients = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})

# Streamlit UI
st.title("Cyber Insurance Premium Estimator")

company_size_input = st.number_input("Company Size (Number of Employees)", min_value=50, max_value=10000, value=500)
industry_risk_input = st.selectbox("Industry Risk Level", [1, 2, 3, 4, 5])
past_incidents_input = st.number_input("Past Cyber Incidents", min_value=0, max_value=10, value=2)
security_measures_input = st.selectbox("Security Measures Rating", [1, 2, 3, 4, 5])
compliance_input = st.selectbox("Compliance Status", [0, 1], format_func=lambda x: "Compliant" if x == 1 else "Non-Compliant")

if st.button("Calculate Premium"):
    input_data = np.array([[company_size_input, industry_risk_input, past_incidents_input, security_measures_input, compliance_input]])
    predicted_premium = model.predict(input_data)[0]
    st.subheader(f"Estimated Premium: ${predicted_premium:,.2f}")
    
    st.subheader("Feature Importance")
    st.write(coefficients)
