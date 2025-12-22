import streamlit as st
import joblib
import numpy as np
import re
from scipy.sparse import hstack, csr_matrix

# ===============================
# Load trained artifacts
# ===============================
clf = joblib.load("classification_model.pkl")
reg = joblib.load("regression_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

scaler = joblib.load("numeric_scaler.pkl")
# Label mapping (same as Phase II)
# label_map = {0: "Easy", 1: "Hard", 2: "Medium"}
label_encoder = joblib.load("label_encoder.pkl")



# ===============================
# Text Cleaning Function
# ===============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9+\-*/=<>^()%.,: ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ===============================
# Feature Engineering
# ===============================
keywords = [
    "graph", "tree", "dp", "dynamic programming", "recursion",
    "greedy", "binary search", "dfs", "bfs",
    "segment tree", "fenwick", "bitmask",
    "matrix", "modulo", "probability",
    "combinatorics", "shortest path"
]


def extract_features(text):
    clean = clean_text(text)

    # TF-IDF
    X_tfidf = tfidf.transform([clean])

    # Hand-crafted features
    text_length = len(clean)
    math_count = len(re.findall(r"[+\-*/=<>^()]", clean))

    keyword_counts = [clean.count(k) for k in keywords]

    numeric_features = np.array(
        [text_length, math_count] + keyword_counts
    ).reshape(1, -1)
    numeric_features = scaler.transform(numeric_features)
    X_numeric = csr_matrix(numeric_features)

    return hstack([X_tfidf, X_numeric])

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="AutoJudge", layout="centered")

st.title("🧠 AutoJudge – Problem Difficulty Predictor")
st.write("Predicts **difficulty class** and **difficulty score** from problem text.")

# Input fields
desc = st.text_area("📘 Problem Description", height=200)
inp = st.text_area("📥 Input Description", height=120)
out = st.text_area("📤 Output Description", height=120)

# Predict button
if st.button("🔍 Predict Difficulty"):
    if desc.strip() == "":
        st.warning("Please enter the problem description.")
    else:
        combined_text = (
            desc + " Input: " + inp + " Output: " + out
        )

        X = extract_features(combined_text)

        # Predictions
        class_pred = clf.predict(X)[0]
        score_pred = reg.predict(X)[0]
        predicted_label = label_encoder.inverse_transform([class_pred])[0]

        # Display results
        st.success("Prediction Successful 🎯")

        st.subheader("📊 Results")
        # st.write(f"**Predicted Difficulty Class:** `{label_map[class_pred]}`")
        st.write(f"**Predicted Difficulty Class:** `{predicted_label}`")
        st.write(f"**Predicted Difficulty Score:** `{round(score_pred, 2)}`")
