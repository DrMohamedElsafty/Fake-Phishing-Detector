import streamlit as st
import joblib
import re
import numpy as np
from scipy.sparse import hstack

# تحميل الموديل والـ vectorizers
model = joblib.load('lgb_model.pkl')
tfidf_w = joblib.load('tfidf_w.pkl')
tfidf_c = joblib.load('tfidf_c.pkl')

def clean_text(text):
    text = re.sub(r"[^A-Za-z0-9\s]", " ", str(text))
    text = text.lower()
    return text

st.title("🚩 Fake Phishing Detector 🚩")
st.write("اكتب أو الصق الرسالة المراد فحصها:")

input_text = st.text_area("رسالتك هنا:")

if st.button("افحص الرسالة"):
    cleaned = clean_text(input_text)
    Xw = tfidf_w.transform([cleaned])
    Xc = tfidf_c.transform([cleaned])
    Xm = np.array([[cleaned.count('http'), len(cleaned)]])
    X = hstack([Xw, Xc, Xm])
    pred = model.predict(X)[0]
    if pred == 1:
        st.error("احتيالي 🚩 (رسالة تصيد أو اختراق)")
    else:
        st.success("سليم ✅ (رسالة آمنة)")
