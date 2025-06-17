import streamlit as st
import pickle
import re

# Load model & vectorizer
with open("model_svm.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Preprocessing fungsi
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Streamlit UI
st.title("Prediksi Minat Penonton Anime Romance")
st.write("Masukkan ulasan, dan model akan memprediksi apakah penonton menyukai anime tersebut.")

user_input = st.text_area("Masukkan ulasan anime di sini:")

if st.button("Prediksi"):
    cleaned = clean_text(user_input)
    X_input = vectorizer.transform([cleaned])
    hasil = model.predict(X_input)[0]

    if hasil == "Suka":
        st.success("✅ Hasil Prediksi: Penonton kemungkinan **Suka** dengan anime ini.")
    else:
        st.error("❌ Hasil Prediksi: Penonton kemungkinan **Tidak Suka** dengan anime ini.")
