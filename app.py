import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="AI Fake Product Detection", page_icon="🛍️", layout="wide")

# -----------------------------
# Demo training data
# -----------------------------
data = pd.DataFrame({
    "price_ratio": [0.95, 1.00, 0.85, 0.35, 0.45, 0.25, 1.10, 0.90, 0.30, 0.50, 1.05, 0.40],
    "rating":      [4.4, 4.7, 4.2, 2.1, 2.8, 1.7, 4.6, 4.1, 2.0, 2.5, 4.8, 2.2],
    "reviews":     [850, 1200, 430, 35, 80, 12, 950, 510, 18, 60, 1500, 25],
    "seller_age":  [48, 60, 36, 2, 4, 1, 72, 30, 1, 3, 80, 2],
    "verified":    [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0],
    "label":       [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0]
})

features = ["price_ratio", "rating", "reviews", "seller_age", "verified"]
X = data[features]
y = data["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_scaled, y)

# -----------------------------
# UI
# -----------------------------
st.title("🛍️ AI-Based Fake Product Detection System")
st.write("A simple machine-learning demo that estimates whether an online product listing looks genuine or suspicious.")

st.info("This is a demo/prototype. It does not prove that a product is genuine or counterfeit.")

st.sidebar.header("Product Details")

price_ratio = st.sidebar.slider(
    "Price compared with typical market price",
    min_value=0.10, max_value=1.50, value=0.80, step=0.05,
    help="1.0 means approximately the normal market price."
)
rating = st.sidebar.slider("Product rating", 0.0, 5.0, 4.0, 0.1)
reviews = st.sidebar.number_input("Number of reviews", 0, 1000000, 250, 10)
seller_age = st.sidebar.number_input("Seller account age (months)", 0, 240, 24, 1)
verified = st.sidebar.selectbox("Seller verified?", ["Yes", "No"])

if st.sidebar.button("🔍 Analyze Product", use_container_width=True):
    verified_value = 1 if verified == "Yes" else 0

    input_df = pd.DataFrame([{
        "price_ratio": price_ratio,
        "rating": rating,
        "reviews": reviews,
        "seller_age": seller_age,
        "verified": verified_value
    }])

    prediction = model.predict(scaler.transform(input_df[features]))[0]
    probability = model.predict_proba(scaler.transform(input_df[features]))[0]
    genuine_probability = probability[1]
    suspicious_probability = probability[0]

    st.subheader("Analysis Result")

    if prediction == 1:
        st.success(f"✅ Product appears **Genuine / Lower Risk**")
    else:
        st.error(f"⚠️ Product appears **Suspicious / Higher Risk**")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Genuine probability", f"{genuine_probability * 100:.1f}%")
    with col2:
        st.metric("Suspicious probability", f"{suspicious_probability * 100:.1f}%")

    st.progress(float(genuine_probability), text="Estimated genuine probability")

    # Explain simple risk factors
    risks = []
    if price_ratio < 0.60:
        risks.append("The price is considerably below the typical market price.")
    if rating < 3.0:
        risks.append("The product rating is relatively low.")
    if reviews < 50:
        risks.append("The listing has relatively few reviews.")
    if seller_age < 6:
        risks.append("The seller account is relatively new.")
    if verified_value == 0:
        risks.append("The seller is not marked as verified.")

    st.subheader("Risk Factors")
    if risks:
        for risk in risks:
            st.write("• " + risk)
    else:
        st.write("No major warning indicators were detected from the entered fields.")

    st.subheader("Entered Product Information")
    st.dataframe(input_df, use_container_width=True)

else:
    st.subheader("How it works")
    st.markdown("""
    1. Enter product and seller information in the sidebar.
    2. The machine-learning model analyzes the features.
    3. The system gives a genuine/suspicious prediction.
    4. It also shows probability and simple risk factors.
    """)

st.caption("AI-Based Fake Product Detection System | Educational prototype")
