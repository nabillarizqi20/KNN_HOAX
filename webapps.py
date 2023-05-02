import streamlit as st
import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

st.set_page_config(page_icon="📰", page_title="Bahan Bakar Minyak", initial_sidebar_state="auto", layout="wide")

hide_menu_style = """
        <style>
        footer {visibility: visible;}
        footer:after{content:'Copyright @ 2023 Nabila Rizqi Amalia'; display:block; position:relative; color:tomato}
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

st.title('KLASIFIKASI BERITA HOAX BAHAN BAKAR MINYAK MENGGUNAKAN ALGORITMA KNN')
st.write('Masukkan tweet untuk dianalisis:')

# Membaca dataset tweet dari file CSV
df = pd.read_csv('tweet.csv')

df.drop('Unnamed: 0', axis=1, inplace=True)
df.drop('Date', axis=1, inplace=True)
df.drop('User', axis=1, inplace=True)

# Menambahkan kolom 'label' dengan nilai default 'non hoax'
df['label'] = 'non hoax'

# strings_to_search = ['PBBM', 'Bahan Bakar Minyak']
# kondisi_hoax = (df['Tweet'].str.contains('|'.join(strings_to_search), case=True))
kondisi_hoax = (df['Tweet'].str.contains('Bahan Bakar Minyak', case=True))
df.loc[kondisi_hoax, 'label'] = 'hoax' 

# Menghilangkan URL dari teks tweet menggunakan regex
df['Tweet'] = df['Tweet'].apply(lambda x: re.sub(r'http\S+', '', x))

# Menghilangkan URL dari teks tweet menggunakan regex
df['Tweet'] = df['Tweet'].apply(lambda x: re.sub(r'@\S+', '', x))

# Melakukan case folding pada teks tweet
df['Tweet'] = df['Tweet'].str.lower()

# Menghapus stopwords pada teks tweet
stop_words = set(stopwords.words('indonesian')) # Menyimpan stopwords dalam set
df['Tweet'] = df['Tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))


# Memisahkan fitur (tweet) dan label (hoax) dari dataset
X = df['Tweet']
y = df['label']

# Menggunakan CountVectorizer untuk mengubah teks tweet menjadi fitur numerik
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.sidebar.subheader("K-Nearest Neighbors")
tetangga = st.sidebar.slider("K", value=5, max_value=10, min_value=0, help="K merupakan tetangga terdekat yang digunakan untuk melakukan klasifikasi data baru")
knn = KNeighborsClassifier(n_neighbors=(tetangga))
knn.fit(X_train, y_train)

# Melakukan prediksi pada data uji
y_pred = knn.predict(X_test)

# Menghitung akurasi model
accuracy = accuracy_score(y_test, y_pred)

# Menghitung jumlah berita hoax dan non-hoax
num_hoax = len(df[df['label'] == 'hoax'])
num_non_hoax = len(df[df['label'] == 'non hoax'])

# Menghitung akurasi untuk berita hoax dan non-hoax
hoax_indices = df[df['label'] == 'hoax'].index
non_hoax_indices = df[df['label'] == 'non hoax'].index
y_pred_hoax = knn.predict(X[hoax_indices])
y_pred_non_hoax = knn.predict(X[non_hoax_indices])
accuracy_hoax = accuracy_score(y[hoax_indices], y_pred_hoax)
accuracy_non_hoax = accuracy_score(y[non_hoax_indices], y_pred_non_hoax)

input_tweet = st.text_area('Input Tweet', '')
if st.button('Analisis'):
    if input_tweet :
        input_tweet = [input_tweet]
        input_tweet = vectorizer.transform(input_tweet)
        prediction = knn.predict(input_tweet)
        if prediction == 'hoax':
            st.write('Hasil Analisis: Berita Hoax')
            st.write('Hasil Akurasi: {:.2f}%'.format(accuracy_hoax * 100))
        else:
            st.write('Hasil Analisis: Non Hoax')
            st.write('Hasil Akurasi : {:.2f}%'.format(accuracy_non_hoax * 100))
    else:
        st.warning('Masukkan tweet terlebih dahulu!')

