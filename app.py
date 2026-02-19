import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Spam Detection", layout="centered")

st.markdown("<h1 style='text-align:center;color:#4CAF50;'>📧 Spam Email Detection</h1>", unsafe_allow_html=True)

# Sample dataset
data = {
    "text": [
        "Win money now",
        "Limited offer claim prize",
        "Meeting at 10am",
        "Project discussion tomorrow",
        "Free lottery ticket",
        "Important business update"
    ],
    "label": [1,1,0,0,1,0]
}

df = pd.DataFrame(data)

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

model = LogisticRegression()
model.fit(X, y)

email = st.text_area("Enter Email Text")

if st.button("Check Spam"):
    transformed = vectorizer.transform([email])
    prediction = model.predict(transformed)

    if prediction[0] == 1:
        st.error("🚨 This is SPAM!")
    else:
        st.success("✅ This is NOT Spam")



streamlit
pandas
scikit-learn
