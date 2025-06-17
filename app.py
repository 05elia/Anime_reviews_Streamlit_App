import streamlit as st
import pickle
import re
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ====== Load Model dan Vectorizer ======
with open("model_svm.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ====== Fungsi Preprocessing Teks ======
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ====== Sidebar: Informasi Penelitian ======
st.sidebar.title("📘 Info Penelitian")
st.sidebar.markdown("""
**Judul:**  
Analisis Minat Penonton Anime Romance Berdasarkan Ulasan

**Peneliti:**  
Elia Jose Alvaro Rahayaan

**Metode:**  
TF-IDF + Support Vector Machine (SVM)

**Akurasi Model:**  
91%

**Dataset:**  
Ulasan pengguna dari MyAnimeList untuk anime bergenre romance
""")

# ====== Halaman Utama ======
st.title("🎌 Prediksi Minat Penonton Anime Romance")
st.write("Masukkan ulasan dari pengguna dan model akan memprediksi apakah mereka menyukai anime tersebut.")

user_input = st.text_area("📝 Masukkan ulasan pengguna:")

if st.button("🎯 Prediksi"):
    if user_input.strip() == "":
        st.warning("Tolong masukkan ulasan terlebih dahulu.")
    else:
        cleaned = clean_text(user_input)
        X_input = vectorizer.transform([cleaned])
        hasil = model.predict(X_input)[0]

        if hasil == "Suka":
            st.success("✅ Hasil Prediksi: Penonton kemungkinan **Suka** dengan anime ini.")
        else:
            st.error("❌ Hasil Prediksi: Penonton kemungkinan **Tidak Suka** dengan anime ini.")

# ====== WordCloud Visualisasi ======
st.markdown("---")
st.subheader("☁️ Visualisasi WordCloud Berdasarkan Label")

try:
    df = pd.read_csv("anime_romance_reviews.csv")

    if 'label' not in df.columns:
        df["skor"] = df["skor"].astype(str).str.extract(r"(\d+)").astype(float)
        df["label"] = df["skor"].apply(lambda x: "Suka" if x >= 7 else "Tidak Suka")

    if "ulasan_clean" not in df.columns:
        df["ulasan_clean"] = df["ulasan"].astype(str).apply(clean_text)

    # Keyword Lists
    negatif_keywords = [
        'boring', 'bad', 'slow', 'disappoint', 'predictable', 'annoying', 'worst',
        'poor', 'waste', 'generic', 'terrible', 'flop', 'hate', 'ugly', 'fail',
        'bland', 'forced', 'cringe', 'awful', 'meh', 'nothing', 'unoriginal',
        'inconsistent', 'confusing', 'cliched', 'stupid', 'unrealistic', 'lame',
        'pointless', 'tedious', 'shallow', 'repetitive', 'frustrating', 'broken',
        'overrated', 'underwhelming', 'forgettable', 'dragging', 'nonsense', 'messy'
    ]

    positif_keywords = [
        'love', 'beautiful', 'amazing', 'great', 'romantic', 'sweet', 'cute', 'emotional',
        'heartwarming', 'masterpiece', 'enjoyed', 'fun', 'good', 'perfect', 'touching',
        'awesome', 'excellent', 'fantastic', 'wonderful', 'brilliant', 'inspiring',
        'relatable', 'unique', 'intense', 'powerful', 'engaging'
    ]

    def filter_kata(text, allowed):
        return ' '.join([w for w in text.split() if w in allowed])

    # WordCloud Positif
    st.write("**WordCloud - Label Suka (Filtered Kata Positif)**")
    df["ulasan_positif_filtered"] = df[df["label"] == "Suka"]["ulasan_clean"].apply(lambda x: filter_kata(x, positif_keywords))
    suka_text = ' '.join(df["ulasan_positif_filtered"].dropna())
    wordcloud_suka = WordCloud(width=800, height=400, background_color='white').generate(suka_text)
    st.image(wordcloud_suka.to_array(), use_column_width=True)

    # WordCloud Negatif
    st.write("**WordCloud - Label Tidak Suka (Filtered Kata Negatif)**")
    df["ulasan_negatif_filtered"] = df[df["label"] == "Tidak Suka"]["ulasan_clean"].apply(lambda x: filter_kata(x, negatif_keywords))
    tidak_suka_text = ' '.join(df["ulasan_negatif_filtered"].dropna())
    wordcloud_tidak = WordCloud(width=800, height=400, background_color='white').generate(tidak_suka_text)
    st.image(wordcloud_tidak.to_array(), use_column_width=True)

except Exception as e:
    st.warning("📂 Dataset tidak ditemukan atau format kolom tidak lengkap.")
    st.text(str(e))
