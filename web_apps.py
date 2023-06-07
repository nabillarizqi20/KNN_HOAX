import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# Baca dataset berita
data = pd.read_csv('dataset berita hoax dan bukan hoax.csv')

# Membuang kolom yang tidak diperlukan
data.drop('Unnamed: 0', axis=1, inplace=True)
data.drop('Teks', axis=1, inplace=True)
data.drop('Rangkuman', axis=1, inplace=True)
data.drop('Penulis', axis=1, inplace=True)

# Menghapus kata "[DISINFORMASI]" dan "[HOAKS]" pada kolom "Judul"
data['Judul'] = data['Judul'].str.replace('\[DISINFORMASI\]', '')
data['Judul'] = data['Judul'].str.replace('\[HOAKS\]', '')

# Melakukan case folding pada teks
data['Judul'] = data['Judul'].str.lower()

# Menghapus stopwords pada teks
stop_words = set(stopwords.words('indonesian'))
data['Judul'] = data['Judul'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Melakukan tokenisasi pada teks tweet
data['Judul'] = data['Judul'].apply(lambda x: word_tokenize(x))

# Melakukan stemming pada teks
stemmer = StemmerFactory().create_stemmer()
data['Judul'] = data['Judul'].apply(lambda x: [stemmer.stem(word) for word in x])

# Menggabungkan token-token menjadi kalimat-kalimat kembali
data['Judul'] = data['Judul'].apply(' '.join)

# Memisahkan fitur dan label
X = data['Judul']
y = data['label']

# Membuat TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model KNN
knn = KNeighborsClassifier(n_neighbors=1)

# Melatih model KNN
knn.fit(X, y)

#Melakukan prediksi pada data uji
y_pred = knn.predict(X_test)

# Menghitung akurasi model
accuracy = accuracy_score(y_test, y_pred)

# Fungsi untuk mendeteksi berita
def detect_hoax(news_text):
    news_text = vectorizer.transform([news_text])
    prediction = knn.predict(news_text)
    return prediction[0]

# Tampilan web menggunakan Streamlit
st.title("Deteksi Berita Hoax BBM dengan Metode KNN")
news_input = st.text_area("Masukkan teks berita:")
if st.button("Deteksi"):
    result = detect_hoax(news_input)
    st.write("Hasil Deteksi: ", result)
    st.write("Akurasi Model: ", accuracy)