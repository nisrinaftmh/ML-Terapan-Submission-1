# **Laporan Proyek Machine Learning Terapan | Prediksi Penyakit Jantung**
###### **Dibuat Oleh : Nisrina Fatimah Parisya**
---
![image](https://github.com/user-attachments/assets/7d93c22e-32e0-464a-8fe1-159cdfadeff8)
---
## **1. Domain Proyek**
### Latar Belakang
Jantung merupakan organ vital yang sangat penting dalam kehidupan semua makhluk hidup, pada kasus ini penyakit jantung merupakan kondisi keadaan jantung seperti gangguan pada fungsi jantung dalam memompa darah yang bisa mengakibatkan disfungsional pemompaan darah ke tubuh. Secara umum terdapat beberapa faktor yang mengakibatkan fungsi jantung diantaranya kolesterol, penyakit bawaan genetik, tekanan darah tinggi, tekanan mental, obesitas dan beberapa faktor lainnya yang bisa menyebabkan gangguan terhadap kualitas hidup seseorang hingga mengakibatkan kematian

Meningkatnya penderita penyakit jantung menimbulkan kekhawatiran sehingga diperlikan adanya upaya pencegahan sejak dini untuk menjaga tubuh agar terhindar dari penyakit tersebut. Peran adanya model machine learning ini dimanfaatkan untuk memprediksi serta menganalisis tanda tanda penyakit jantung berdasaran data pasien serta mengidentifikasi pola pola tertentu yang dapat menimbulkan risiko penyakit jantung.

Proyek ini tentunya memiliki relevansi dalam bidang kesehatan dengan membantu tenaga mendis maupun pasien untuk mendeteksi adanya potensi penyakit jantung dengan akurat. Sistem prediksi ini diharapkan dapat memberikan penjelasan kepada pasien terkait kesehatan jantungnya dan memingkinkan untuk membantu pengawasan medis untuk menghindari adanya terkena penyakitjantung ataupun mencegah adanya perkembangan penyakit jantung yang lebih parah

Manfaat yang didapat untuk tenaga medis diantaranya :
1.  Membantu mempercepat proses _screening_ pasien yang kemungkinan beresiko memiliki penyakit jantung
2.  Melakukan monitoring terkait dengan rekam medis pasien secara berkala

Manfaat yang dapat dirasakan pasien diantaranya:
1. Mendeteksi dini risiko terkena serangan jantung serta memperhatikan pola hidup sejak awal untuk pencegahan
2. Pasien dapat mendapat akses informasi yang lebih cepat dan mudah tanpa harus pemeriksaan yang mahal.

## **2. Business Understanding**
## Problem Statements
1. Bagaimana cara mengembangkan model machine learning yang dapat memprediksi risiko penyakit jantung pada pasien berdasarkan data medis?
2. Bagaimana Hasil Prediksi yang dilakukan? Permodelan apa yang paling efektif untuk digunakan?

## Goals
1. Membuat model machine learning yang dapat memprediksi risiko penyakit jantung pada pasien
2. Menentukan permodelan yang cocok untuk analisis prediksi penyakit jantung

## Solution Statement
Terdapat beberapa tahapan yang diperlukan untuk mencapai tujuan proyek tersebut diantaranya
1. Melakukan pemahaman data, melakukan pembersihan data dan melihat korelasi antar variabel.
2. Membuat permodelan Machine Learning untuk dataset yang digunakan dengan menggunakan permodelan Random Forest, Logistic Regression dan K-Nearest Neighbour.
3. Menerapkan StandardSCaler dalam menormalisasi data agar meningkatkan akurasi model
4. Melakukan Evaluasi dengan menggunakan metrik Akurasi, Precision, Recall dan F1 Score

## **3. Data Understanding**
Dataset yang digunakan merupakan dataset berjudul "Common Heart Disease (4 Hospitals)  yang dapat diakses melalui kaggle dengan link berikut ini [Common Heart Disease Dataset]( https://www.kaggle.com/datasets/denysskyrda/common-heart-disease-data-4-hospitals). Dataset ini terdiri dari 920 Baris dengan data pasien dengan parameter berupa variabel yang relevan dengan penyakit jantung.

**Kondisi Data**
* Tidak ditemukan missing value pada 920 baris dan 15 kolom data.
* Ditemukan 2 data duplikat yang telah dihapus.

**Insight**
- Korelasi tinggi ditemukan antara cp (nyeri dada), exang (nyeri dada saat olahraga), oldpeak, thal, dan ca dengan target variable, mengindikasikan fitur penting untuk prediksi.
- Kolom Source Tag tidak digunakan dalam pemodelan karena hanya menunjukkan asal rumah sakit data.
  
### Variabel-variabel pada Dataset
Berikut adalah Fitur yang terdapat dalam dataset:
| No. | Nama Kolom      | Deskripsi                                                                                         | Nilai/Contoh                                            |
|-------|-----------------|---------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| 1     | age             | Usia pasien dalam tahun                                                                           | Contoh: 55                                             |
| 2     | sex             | Jenis kelamin pasien                                                                            | 1 = laki-laki, 0 = perempuan                            |
| 3     | cp              | Tipe nyeri dada                                                                                 | 0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic |
| 4     | trestbps        | Tekanan darah istirahat dalam mm Hg                                                              | Contoh: 130                                            |
| 5     | chol            | Kolesterol serum dalam mg/dl                                                                      | Contoh: 250                                            |
| 6     | fbs             | Gula darah puasa > 120 mg/dl                                                                     | 1 = benar, 0 = salah                                  |
| 7     | restecg        | Hasil elektrokardiografi istirahat                                                               | 0: Normal, 1: Abnormalitas ST-T, 2: Hipertrofi ventrikel kiri |
| 8     | thalach         | Detak jantung maksimum yang dicapai                                                               | Contoh: 150                                            |
| 9     | exang           | Angina yang dipicu oleh olahraga                                                                   | 1 = ya, 0 = tidak                                   |
| 10    | oldpeak         | Depresi ST yang diakibatkan oleh olahraga relatif terhadap istirahat                                | Contoh: 1.5                                          |
| 11    | slope           | Kemiringan segmen ST pada puncak olahraga                                                          | 0: Upsloping, 1: Flat, 2: Downsloping             |
| 12    | ca              | Jumlah pembuluh darah utama yang diwarnai oleh fluoroskopi                                         | 0-3                                                   |
| 13    | thal            | Kelainan thalassemia                                                                               | 0: Normal, 1: Fixed defect, 2: Reversible defect  |
| 14    | target          | Diagnosis penyakit jantung                                                                         | 1 = penyakit jantung, 0 = tidak ada penyakit jantung |
| 15    | Source Tag        | Data ini hanya berisi nama rumah sakit data pasien diambil, namun dalam pengerjaan model kita tidak menggunakan data ini | 

Pada semua kolom tersebut terdapat 13 kolom dengan tipe data float, 1 kolom dengan tipe data int dan 1 kolom dengan data kategorikal. Pada pembuatan model ini tidak digunakan data _Sourge Tag_ dengan alasan fokus utama prediksi hanya untuk menganalisis adanya penyakit jantung atau tidak pada pasien.


### Exploratory Data Analysis (EDA)
### Cek Ukuran Data
Tahapan ini dilakukan untuk memahami isi dataset. hal pertama yang dilakukan adalah memahami dan mengecek isi dari dataset dengan menggunakan `.shape', .info() `dan `.describe()`
``` df.shape```

![image](https://github.com/user-attachments/assets/56b97b53-6144-442c-973f-cff37ec4b97a)


Output pada kode tersebut menunjukkan bahwa dataset terdiri dari 920 baris dan 15 kolom

### Cek Informasi Data
Disini kita akan melakukan pengecekan tipe data dan melihat informasi dataset dengan fungsi `.info`

![image](https://github.com/user-attachments/assets/89a735e5-9fc6-4d28-ab4f-2a916c4416c5)

```
# Melihat info dataset untuk mengetahui tipe data setiap kolom
df.info()
```

### Cek Missing Value
pada tahapan ini kita dapat melakukan pengecekan missing value dalam dataset tersebut diantaranya menggunakan funsgi .`isnull().sum()` untuk mengetahui missing value di setiap kolom
![image](https://github.com/user-attachments/assets/a5eff07a-8f93-4d93-b740-df64a41add98)
Berdasarkan output diatas tidak ditemukan adanya missing value, maka dari itu kita tidak perlu melakukan eksekusi drop atau mengisi nilai NaN pada data karena semua data terisi dengan baik sehingga tidak diperlukan adanya penanganan missing value.

### Cek Duplicate Data
Pada tahapan ini kita dapat melakukan pengecekan duplikasi data dengan df.`duplicated().sum(). `Setelah melakukan pengecekan ternyata terdapat data yang mengalami duplikasi 

![image](https://github.com/user-attachments/assets/0fbd84fe-66fe-4500-a839-15de82b1f621)

```
# cek duplikasi data
print("Cek Duplikasi Data:")
print(f"Jumlah duplikasi data sebelum dihapus: {df.duplicated().sum()}")
```

### Data Distribution
![image](https://github.com/user-attachments/assets/5b9c0d2a-bf9c-4f4a-a6f0-380da6d9a871)

Hasil Correlation Map diatas beberikan beberapa informasi yaitu :
*   Fitur yang sangat berkorelasi dengan label target adalah cp (nyeri dada), exang (nyeri dada saat berolahraga), oldpeak, thal (hasil tes thalassemia) dan ca (jumlah pembuluh darah).
*   Pada hasil tersebut dapat dilihat bahwa jenis kelamin laki laki (sex=1) memiliki resiko penyakit jantung yang lebih tinggi dibandingkan perempuan.


## **4. Data Preparation**
#### Pembersihan Fitur
Berikutnya saya akan menghapus kolom source yang berisi nama RS data penyakit ini diambil, alasannya karena fokus saya disini hanya untuk kebutuhan analisis diagnosis
```df = df.drop('source', axis=1)```

#### Handling Duplicate Data
Sebagaimana setelah melakukan pengecekan data dapat kita lihat bahwa ada 2 data yang mengalami duplikasi sehingga diperlukan adanya penghapusan data duplikat menggunakan fungsi `.drop_duplicates()`

![image](https://github.com/user-attachments/assets/52c5c2dd-cd44-48cd-bdfb-fd41e78649c5)


``` 
# Hapus duplikasi data
df = df.drop_duplicates()

print(f"Jumlah duplikasi data setelah dihapus: {df.duplicated().sum()}")

# Cek ukuran data setelah menghapus duplikasi
print(f"Jumlah baris data setelah menghapus duplikasi: {len(df)}")

```

### Membagi pengelompokan jenis data pada fitur
pada tahapan ini saya mengecek tipe data terhadap kolom yang akan di gunakan untuk permodelan, disini saya menggunakan `df.select_dtypes(include=['number']).columns.tolist()` dan `df.select_dtypes(include=['object']).columns.tolist()`. dapat dilihat pada output dibawah kode tersebut apa saja kolom dengan tipe data numerik. dan untuk kolom kategorikal kosong karena pada dataset memang hanya membutuhkan kolom numerikal untuk proses

```
# Mengecek tipe data setiap kolom dan mengelompokkannya
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print("Kolom Numerikal:")
print(numerical_cols)
print("\nKolom Kategorikal:")
categorical_cols
```

Output :

![image](https://github.com/user-attachments/assets/dc3292eb-f35e-4a06-8fc3-6a2f412c845a)


### Data Splitting
Tahapan ini merupakan tahapan sebelum memasuki permodelan. Langkah yang saya lakukan diantaranya :
* Karena data yang saya gunakan disini sudah berformat numerik dan sudah terdapat label pada data maka saya tidak melakukan Label Encoding.
* Disini saya membagi dataset menjadi data latih (train) sebesar 80% dan data uji (testing) sebesar 20%
```bash  
# Bagi dataset menjadi training (80%) dan testing (20%)
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Cek ukuran data
print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")
```
![image](https://github.com/user-attachments/assets/e0dc5028-7ad7-492d-b500-0a541e2729ab)

## Normalisasi Data
Langkah kedua pada tahapan ini saya melakukan normalisasi menggunakan `StandardScaler()` dengan tujuan meningkatkan hasil prediksi lebih akurat dan stabil
```bash  
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
X_train = scaler.fit_transform(X_train)  
X_test = scaler.transform(X_test)
```


## **5. Modelling**
Algoritma yang dipilih dan digunakan untuk proyek ini ditekankan dengan algoritma klasifikasi Random Forest, Logistic Regression dan K-Nearest Neighbour untuk melakukan perbandingan mana performa model yang lebih baik untuk memprediksi dataset ini

### 1. Random Forest
#### Cara Kerja

Random Forest adalah algoritma ensemble learning yang menggabungkan multiple decision trees untuk membuat prediksi yang lebih akurat dan stabil. Algoritma ini bekerja dengan cara:
- Bootstrap Sampling: Membuat beberapa subset data training secara random dengan replacement
- Feature Randomness: Pada setiap node split, hanya subset random dari features yang dipertimbangkan  
- Tree Construction: Membangun decision tree untuk setiap subset data
- Voting Mechanism: Untuk klasifikasi, menggunakan majority voting dari semua trees

#### Parameter:

- random_state=42
- n_estimators=100 (default)
- max_depth=None (default)
- min_samples_split=2 (default)
- min_samples_leaf=1 (default)

#### Tahapan Penyusunan Model:

1. Dataset dibagi menjadi data latih dan uji (X_train, X_test, y_train, y_test).
2. Model RandomForestClassifier diinisialisasi dengan parameter random_state=42.
3. Model dilatih pada X_train dan y_train menggunakan metode fit().
4. Prediksi dilakukan pada X_test menggunakan metode predict().
5. Evaluasi dilakukan dengan membandingkan y_pred_dt dengan y_test.

#### Kelebihan:

- Mengurangi overfitting dibanding pohon keputusan tunggal
- Mampu menangani data non-linear
- Menyediakan informasi feature importance
- Robust terhadap outliers dan noise

#### Kekurangan:

- Lebih lambat dibanding model sederhana 
- Sulit untuk interpretasi secara keseluruhan
- Membutuhkan lebih banyak memori

#### Implementasi:

- Model dilatih langsung pada data asli tanpa normalisasi karena Random Forest tidak sensitif terhadap skala fitur
- Parameter random_state=42 digunakan untuk memastikan hasil yang konsisten dan dapat direproduksi
- Model RandomForestClassifier diinisialisasi dan dilatih menggunakan data latih (X_train, y_train)
- Model digunakan untuk memprediksi label pada data uji (X_test)

```bash  
from sklearn.ensemble import RandomForestClassifier
# Inisialisasi dan latih model Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Prediksi
y_pred_dt = rf_model.predict(X_test)
```

### 2. Logistic Regression
#### Cara Kerja

Logistic Regression adalah algoritma linear classifier yang menggunakan fungsi logistic (sigmoid) untuk memetakan nilai real ke probabilitas antara 0 dan 1. Cara kerja algoritma:
- Linear Combination: Menghitung kombinasi linear dari input features: z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
- Sigmoid Function: Mengaplikasikan fungsi sigmoid: σ(z) = 1/(1 + e^(-z))
- Probability Mapping: Mengkonversi output ke probabilitas kelas
- Decision Boundary: Menggunakan threshold (biasanya 0.5) untuk klasifikasi final

#### Parameter:

- solver='lbfgs' (default)
- max_iter=100 (default)
- C=1.0 (default)
- penalty='l2' (default)
- random_state=42

#### Tahapan Penyusunan Model:

1. Dataset dibagi menjadi data latih dan uji (X_train, X_test, y_train, y_test).
2. Model LogisticRegression diinisialisasi dengan parameter random_state=42.
3. Model dilatih pada X_train dan y_train menggunakan metode fit().
4. Prediksi dilakukan pada X_test menggunakan metode predict().
5. Evaluasi dilakukan dengan membandingkan hasil prediksi dengan y_test.

#### Kelebihan:

- Sederhana dan cepat dalam training dan prediksi
- Memberikan probabilitas output yang dapat diinterpretasi
- Tidak memerlukan tuning parameter yang kompleks
- Efisien untuk dataset besar
- Memberikan koefisien yang dapat diinterpretasi

#### Kekurangan:

- Hanya cocok untuk masalah yang linearly separable
- Sensitif terhadap outliers
- Memerlukan feature scaling untuk performa optimal
- Tidak dapat menangkap hubungan non-linear tanpa feature engineering

#### Implementasi:

- Model menggunakan solver 'lbfgs' yang cocok untuk dataset kecil hingga menengah.
- Parameter C=1.0 digunakan untuk mengontrol kekuatan regularisasi, di mana nilai yang lebih kecil berarti regularisasi yang lebih kuat.
- Parameter random_state=42 digunakan untuk memastikan hasil yang konsisten dan dapat direproduksi.
- Model LogisticRegression diinisialisasi dan dilatih menggunakan data latih (X_train, y_train).
- Model digunakan untuk memprediksi label pada data uji (X_test).
- Menggunakan random_state=42 untuk memastikan reproducibility hasil

```bash  
from sklearn.linear_model import LogisticRegression
# Inisialisasi dan latih model Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)

# Prediksi
y_pred_lr = lr_model.predict(X_test)
```
### 3. K-Nearest Neighbour
#### Cara Kerja

KNN adalah algoritma lazy learning yang melakukan klasifikasi berdasarkan kedekatan dengan data training. Proses kerja algoritma:
- Distance Calculation: Menghitung jarak antara data test dengan semua data training
- Neighbor Selection: Memilih k tetangga terdekat berdasarkan jarak yang dihitung
- Majority Voting: Menggunakan voting mayoritas dari k tetangga untuk menentukan kelas
- Tie Breaking: Menangani kasus seri dengan aturan tertentu

#### Parameter:
- n_neighbors=5 (default)
- weights='uniform' (default)
- algorithm='auto' (default)
- metric='minkowski' (default)
- p=2 (default)

#### Tahapan Penyusunan Model:

1. Data fitur (X) dinormalisasi menggunakan StandardScaler karena KNN sensitif terhadap skala data.
2. Dataset dibagi menjadi data latih dan uji (X_train, X_test, y_train, y_test).
3. Model KNeighborsClassifier diinisialisasi dengan parameter default.
4. Model dilatih pada X_train_scaled dan y_train.
5. Prediksi dilakukan pada X_test_scaled dan evaluasi dengan y_test.

#### Kelebihan:

- Sederhana dan mudah diimplementasikan
- Tidak memerlukan asumsi tentang distribusi data
- Baik untuk menangkap hubungan lokal antar data
- Efektif untuk dataset dengan pola yang kompleks

#### Kekurangan:

- Sensitif terhadap skala data (diperlukan normalisasi)
- Performa menurun jika jumlah data besar atau berdimensi tinggi
- Tidak memberikan interpretasi fitur
- Komputasi prediksi lambat untuk dataset besar

#### Implementasi:

- Model KNeighborsClassifier digunakan dengan parameter default, yaitu n_neighbors=5 dan metrik jarak Euclidean.
- Model dilatih menggunakan data latih X_train dan y_train.
- Prediksi dilakukan pada data uji X_test dan hasilnya disimpan dalam y_pred_knn.

```bash  
from sklearn.neighbors import KNeighborsClassifier
# Inisialisasi dan latih model KNN
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

# Prediksi
y_pred_knn = knn_model.predict(X_test)
```
Pemilihan Model Terbaik
Berdasarkan hasil evaluasi, ketiga model memberikan hasil performa yang baik dengan akurasi di atas 80%. Perbandingan performa model adalah sebagai berikut:
| Model                     | Accuracy | F1-Score |
|---------------------------|----------|----------|
| Random Forest             | 0.8423   | 0.8414   |
| Logistic Regression       | 0.8206   | 0.8199   |
| K-Nearest Neighbors (KNN) | 0.8423   | 0.8414   |


## **6. Evaluation**

### Evaluasi Hasil Modelling

Proyek ini akan dievaluasi menggunakan beberapa metrik untuk menilai model klasifikasi, diantaranya :

**1. Accuracy (Akurasi)** : Metrik ini mengukur proporsi prediksi dari total prediksi

**2. F1-Score** : Metrik ini mengukur rata rata dari precission dan recal untuk melohat keseimbangan antara hasil false positive dan false negative dari hasil permodelan

Berdasarkan hasil _classification report _ diperoleh hasil dari Accuracy dan F1-Score pada masing masing model dilampirkan dibawah ini

<img src="https://github.com/user-attachments/assets/88b8805a-779a-494d-8cb0-314d56c7f4c3
" width="300"/>
<img src="https://github.com/user-attachments/assets/0cdf58da-7b0f-4ec3-b9b2-090ad1ef4871
" width="300"/>
<img src="https://github.com/user-attachments/assets/dc3b1703-36a3-46f4-a962-ff5be1b4a6b9
" width="300"/>

**1. Random Forest**
* Memiliki akurasi sebesar 84,23 % dan weighted F1-score sebesar 0.8423, menunjukkan performa yang stabil dan baik.
* Dapat memprediksi dengan baik pada kelas mayoritas (pasien sakit), dengan recall tinggi sebesar 0.89 untuk kelas 1.


**2. Logistic Regression**
* Mencapai akurasi sebesar 82.06% dan weighted F1-score sebesar 0.82065
* Memiliki recall tertinggi (0.86) untuk kelas 1 (pasien sakit), menunjukkan kemampuan yang sangat baik dalam mendeteksi pasien sakit.
* Model ini tergolong sederhana dan interpretable, namun tetap memberikan hasil prediksi yang sangat baik.

**3. K-Nearest Neighbors (KNN)**
* Memberikan akurasi sebesar 84,2% dan weighted F1-score sebesar 0.842, yang setara dnegan hasil random forest.
* Recall yang tinggi (0.89) untuk kelas 1 membuatnya cocok untuk kasus di mana deteksi positif sangat penting.

### Evaluasi Bisnis

#### Problem Solving
1. **Bagaimana cara mengembangkan model machine learning yang dapat memprediksi risiko penyakit jantung?**  
   Problem berhasil dijawab melalui pengembangan dan evaluasi tiga model klasifikasi:
   - **Random Forest** mencapai akurasi 84% dan F1-score 0.842
   - **Logistic Regression** akurasi 83% dan F1-score 0.820
   - **K-Nearest Neighbors (KNN)** akurasi 84% dan F1-score 0.842

   Ketiganya menunjukkan performa yang baik, dan proses mencakup:
   - Praproses data secara menyeluruh
   - Pemilihan algoritma yang relevan
   - Evaluasi metrik klasifikasi (precision, recall, F1-score)


2. **Bagaimana Hasil Prediksi yang dilakukan? Permodelan apa yang paling efektif untuk digunakan?**  
   Telah dijawab melalui:
   - Evaluasi hasil menunjukkan bahwa random forest dan knn memiliki tingkat akurasi dan skor f1 paling tinggi dan stabil


### Capaian Goals

1. **Model prediksi risiko penyakit jantung**  
   Tercapai. Model Random Forest dan KNN menjadi pilihan terbaik dari sisi akurasi dan F1-score. Logistic Regression menjadi alternatif baik karena recall yang tinggi dan interpretabilitas.

2. **Identifikasi faktor risiko penyakit jantung**  
   Tercapai. Insight dari EDA dan analisis fitur menunjukkan keterkaitan kuat antara faktor medis (seperti chol, cp, thalach, dan age) dengan risiko penyakit jantung.

### Dampak dari Solusi yang Dirancang

- **Evaluasi multi-model** memberikan dasar kuat untuk pemilihan model terbaik
- **Random Forest** dan **KNN** unggul dari sisi akurasi dan generalisasi
- **Random Forest** dan **KNN** menunjukkan recall tinggi (0.89), ideal untuk aplikasi medis di mana deteksi positif lebih diutamakan
- **Logistic Regression** juga memiliki angka recall yang cukup ideal di angka 0,86


## **7. Kesimpulan**

Model yang dikembangkan telah berhasil menjawab seluruh problem statement dan mencapai goals yang ditetapkan. Dari keseluruhan model, **Random Forest** dan **KNN** merupakan model terbaik untuk diterapkan pada dataset ini karena memiliki akurasi dan F1-Score tertinggi, yaitu Accuracy sebesar **0.8423** dan F1-Score sebesar **0.8414**.

Secara keseluruhan, evaluasi menunjukkan bahwa solusi yang diimplementasikan berdampak positif terhadap kualitas prediksi dan akurasi sistem deteksi dini penyakit jantung, serta layak digunakan sebagai alat bantu dalam proses **screening penyakit jantung** di bidang medis.


## **Referensi**
Rahmada, A. ., & Susanto, E. R. (2025). Peningkatan Akurasi Prediksi Penyakit Jantung dengan Teknik SMOTEENN pada Algoritma Random Forest. _Jurnal Pendidikan Dan Teknologi Indonesia_, 4(12), 795-803. https://doi.org/10.52436/1.jpti.524

S. A. T. Al Azhima, D. Darmawan, N. F. Arief Hakim, I. Kustiawan, M. Al Qibtiya, dan N. S. Syafei, “Hybrid Machine Learning Model untuk memprediksi Penyakit Jantung dengan Metode Logistic Regression dan Random Forest”, j. teknologi terpadu, vol. 8, no. 1, hlm. 40–46, Jul 2022.

