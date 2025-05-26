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
2. Faktor-faktor apa saja yang paling berkorelasi dengan penyakit jantung berdasarkan analisis dataset?

## Goals
1. Membuat model machikne learning yang dapat memprediksi risiko penyakit jantung pada pasien
2. Melakukan identifikasi fator yang mempengaruhi terjadinya penyakit jantung

## Solution Statement
Terdapat beberapa tahapan yang diperlukan untuk mencapai tujuan proyek tersebut diantaranya
1. Melakukan pemahaman data dan mengeksplorasi data dengan melihat potensi outlier, melakukan pembersihan data dan melihat korelasi antar variabel.
2. Membuat permodelan Machine Learning untuk dataset yang digunakan dengan menggunakan permodelan Random Forest, Logistic Regression dan K-Nearest Neighbour.
3. Menerapkan StandardSCaler dalam menormalisasi data agar meningkatkan akurasi model
4. Melakukan Evaluasi dengan menggunakan metrik Akurasi, Precision, Recall dan F1 Score

## **3. Data Understanding**
Dataset yang digunakan merupakan dataset berjudul "Common Heart Disease (4 Hospitals)  yang dapat diakses melalui kaggle dengan link berikut ini [Common Heart Disease Dataset]( https://www.kaggle.com/datasets/denysskyrda/common-heart-disease-data-4-hospitals). Dataset ini terdiri dari 920 Baris dengan data pasien dengan parameter berupa variabel yang relevan dengan penyakit jantung.

### Variabel-variabel pada Dataset
Berikut adalah variabel-variabel yang terdapat dalam dataset:
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

### Handling Missing Value
pada tahapan ini kita dapat melakukan pengecekan missing value dalam dataset tersebut diantaranya menggunakan funsgi .`isnull().sum()` untuk mengetahui missing value di setiap kolom
![image](https://github.com/user-attachments/assets/a5eff07a-8f93-4d93-b740-df64a41add98)
Berdasarkan output diatas tidak ditemukan adanya missing value, maka dari itu kita tidak perlu melakukan eksekusi drop atau mengisi nilai NaN pada data karena semua data terisi dengan baik sehingga tidak diperlukan adanya penanganan missing value.

### Handling Outlier
Saat melakukan pengecekan statistik deskriptif pada dataset, terdeteksi indikasi  adanya outlier pada beberapa variabel, terutama pada trestbps (tekanan darah), chol (kolesterol), dan oldpeak. Dalam menangani outlier, disini menggunakan metode IQR+Median Replacement untuk mempertahankan jumlah data.
<p>
  <img src="https://github.com/user-attachments/assets/36271e84-a488-495b-9d9e-5077c0333ff9" width="150" />
  <img src="https://github.com/user-attachments/assets/c833b939-755f-4d55-a43d-e2f79c5ecfcc" width="150" />
</p>
dari yang kita lihat diatas ada beberapa kemungkinan data yang mengalami outlier diantaranya
*   trestbps (tekanan darah) : alasannya karena tekanan darah di nilai 0 pada kolom min sangat tidak normal dalam medis
*   chol (kolesterol) : alasannya karena kolesterol berada di nilai 0 pada kolom min sangat tidak normal dalam medis
Setelah Outlier ditangani data dianggap sudah terdistribusi dengan lebih baik

![image](https://github.com/user-attachments/assets/d7005e9f-5428-4b38-9a9d-256a297f4af4)

### Handling Duplicate Data
Pada tahapan ini kita dapat melakukan pengecekan duplikasi data dengan df.`duplicated().sum(). `Setelah melakukan pengecekan ternyata terdapat data yang mengalami duplikasi sebanyak 2 duplikasi sehingga program menghapus data yang duplikat dengan` .drop_duplicates()` dan jumlah data sekatang menjadi 918 baris.

### Data Distribution
Berdasarkan grafik tersebut dapat kita lihat maypritas dari data numerik menunjukkan data yang terdistribusi mendekati normal 
![image](https://github.com/user-attachments/assets/2a265610-3697-49ac-a244-1a66504d80a6)

![image](https://github.com/user-attachments/assets/5b9c0d2a-bf9c-4f4a-a6f0-380da6d9a871)
Hasil Correlation Map diatas beberikan beberapa informasi yaitu :
*   Fitur yang sangat berkorelasi dengan label target adalah cp (nyeri dada), exang (nyeri dada saat berolahraga), oldpeak, thal (hasil tes thalassemia) dan ca (jumlah pembuluh darah).
*   Pada hasil tersebut dapat dilihat bahwa jenis kelamin laki laki (sex=1) memiliki resiko penyakit jantung yang lebih tinggi dibandingkan perempuan.


## **4. Data Preparation**
### Data Splitting
Dataset tersebut dibagi menjadi data latih (train) dan data uji (testing) dengan proporsi 80:20. Hasil pembagian menghasilkan 734 data training dan 184 data testing.
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

#### 1. Random Forest
Pada permodelan ini yang saya lakukan adalah :
* Memanfaatkan library scikit-learn dan
*   Menggunakan fungsi `RandomForestClassifier()`
* Parameteer yang digunakan adalah `random_state=42` yang artinya membuat model dengan random seed agar hasil yang diharapkan bisa konsiste
```bash  
from sklearn.ensemble import RandomForestClassifier
# Inisialisasi dan latih model Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Prediksi
y_pred_dt = rf_model.predict(X_test)
```
#### 2. Logistic Regression
Pada permodelan ini yang saya lakukan adalah :
* Memanfaatkan library scikit-learn dan
*   Menggunakan fungsi `LogisticRegression()`
* Parameter yang digunakan adalah random_state=42 yang artinya membuat model dengan random seed agar hasil yang diharapkan bisa konsisten
```bash  
from sklearn.linear_model import LogisticRegression
# Inisialisasi dan latih model Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)

# Prediksi
y_pred_lr = lr_model.predict(X_test)
```
#### 3. K-Nearest Neighbour
Pada permodelan ini yang saya lakukan adalah :
* Memanfaatkan library scikit-learn dan
*   Menggunakan fungsi `KNeighborsClassifier() dengan parameter default yang nantinya disimpan ke variabel knn_model
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
| Random Forest             | 0.8369   | 0.8364   |
| Logistic Regression       | 0.8315   | 0.8308   |
| K-Nearest Neighbors (KNN) | 0.8152   | 0.8145   |


## **6. Evaluation**
Proyek ini akan dievaluasi menggunakan beberapa metrik untuk menilai model klasifikasi, diantaranya :
**1. Accuracy (Akurasi)** : Metrik ini mengukur proporsi prediksi dari total prediksi
**2. F1-Score** : Metrik ini mengukur rata rata dari precission dan recal untuk melohat keseimbangan antara hasil false positive dan false negative dari hasil permodelan
**3. Confusion Matrix** : Metrik ini memberikan gambaran detail tentang jumlah True Positive (TP), False Positive (FP), True Negative (TN), dan False Negative (FN). Evaluasi ini penting untuk memahami jenis kesalahan yang dilakukan model.

Berdasarkan hasil _classification report _ diperoleh hasil dari Accuracy dan F1-Score pada masing masing model dilampirkan dibawah ini
<img src="https://github.com/user-attachments/assets/7dfc6659-9b27-404a-a95f-df9c5000c002" width="300"/>
<img src="https://github.com/user-attachments/assets/4e871694-cd8d-4a1b-8f90-d95778de3d25" width="300"/>
<img src="https://github.com/user-attachments/assets/249bd675-17c3-4dc2-877d-b46c565a6ec5" width="300"/>

Berdasarkan hasil Confussion Matrix diperoleh kesimpulan berikut ini :
![image](https://github.com/user-attachments/assets/21c9e37d-a3e7-4dc0-8032-ed3d4fff3c88)
1.Random Forest dan Logistic Regression memiliki hasil confusion matrix yang identik, menunjukkan pola prediksi yang sama:
* 88 pasien dengan penyakit jantung terdeteksi dengan benar (TP)
* 63 pasien tanpa penyakit jantung terdeteksi dengan benar (TN)
* 14 pasien dengan penyakit jantung tidak terdeteksi (FN)
* 19 pasien tanpa penyakit jantung diidentifikasi memiliki penyakit (FP)


2. KNN menunjukkan performa yang lebih baik dalam beberapa aspek:
* True Positive meningkat menjadi 91 (berhasil mendeteksi 3 pasien sakit lebih banyak)
* False Negative berkurang dari 14 menjadi 11 (lebih sedikit pasien sakit yang tidak terdeteksi)
* False Positive sedikit berkurang dari 19 menjadi 18.

## Evaluation: Model Performance vs Business Understanding

### Hubungan dengan Problem Statements

1. **Bagaimana cara mengembangkan model machine learning yang dapat memprediksi risiko penyakit jantung?**  
   Problem berhasil dijawab melalui pengembangan dan evaluasi tiga model klasifikasi:
   - **Random Forest** mencapai akurasi 84% dan F1-score 0.84
   - **Logistic Regression** akurasi 83% dan F1-score 0.83
   - **K-Nearest Neighbors (KNN)** akurasi 82% dan F1-score 0.81

   Ketiganya menunjukkan performa yang baik, dan proses mencakup:
   - Praproses data secara menyeluruh
   - Pemilihan algoritma yang relevan
   - Evaluasi metrik klasifikasi (precision, recall, F1-score)
   - Analisis Confusion Matrix untuk interpretasi prediksi model

2. **Faktor-faktor apa saja yang paling berkorelasi dengan penyakit jantung?**  
   Telah dijawab melalui:
   - Analisis korelasi antar fitur
   - Visualisasi distribusi data
   - Feature importance dari Random Forest yang mengidentifikasi fitur-fitur dominan dalam prediksi penyakit jantung

### Capaian Goals

1. **Model prediksi risiko penyakit jantung**  
   Tercapai. Model Random Forest menjadi pilihan terbaik dari sisi akurasi dan F1-score. Logistic Regression menjadi alternatif baik karena recall yang tinggi dan interpretabilitas.

2. **Identifikasi faktor risiko penyakit jantung**  
   Tercapai. Insight dari EDA dan analisis fitur menunjukkan keterkaitan kuat antara faktor medis (seperti chol, cp, thalach, dan age) dengan risiko penyakit jantung.

### Dampak dari Solusi yang Dirancang

- **Evaluasi multi-model** memberikan dasar kuat untuk pemilihan model terbaik
- **Random Forest** unggul dari sisi akurasi dan generalisasi
- **Logistic Regression** menunjukkan recall tinggi (0.87), ideal untuk aplikasi medis di mana deteksi positif lebih diutamakan
- **KNN** menunjukkan peningkatan pada TP (91 pasien berhasil dideteksi) dan pengurangan FP dibanding model lainnya

### Insight dari Confusion Matrix

- **Random Forest & Logistic Regression**:
  - 88 pasien sakit terdeteksi dengan benar (True Positive)
  - 63 pasien sehat terdeteksi dengan benar (True Negative)
  - 14 pasien sakit tidak terdeteksi (False Negative)
  - 19 pasien sehat salah diklasifikasikan sebagai sakit (False Positive)

- **KNN**:
  - TP meningkat menjadi 91, artinya mendeteksi 3 pasien sakit lebih banyak
  - FP menurun dari 19 menjadi 18, sehingga kesalahan identifikasi pasien sehat juga berkurang

## **7. Kesimpulan**

Model yang dikembangkan telah berhasil menjawab seluruh problem statement dan mencapai goals yang ditetapkan. Dari keseluruhan model, **Random Forest** merupakan model terbaik untuk diterapkan pada dataset ini karena memiliki akurasi dan F1-Score tertinggi, yaitu Accuracy sebesar **0.8369** dan F1-Score sebesar **0.8364**. 

Namun, jika dilihat dari hasil **Confusion Matrix**, model **K-Nearest Neighbors (KNN)** dapat dipertimbangkan karena mampu meminimalkan jumlah **False Negative**, yaitu kasus ketika pasien sakit tidak terdeteksi. Hal ini penting dalam konteks medis untuk mengurangi risiko kelalaian dalam mendeteksi pasien yang benar-benar sakit.

Secara keseluruhan, evaluasi menunjukkan bahwa solusi yang diimplementasikan berdampak positif terhadap kualitas prediksi dan akurasi sistem deteksi dini penyakit jantung, serta layak digunakan sebagai alat bantu dalam proses **screening awal** di bidang medis.



## **Referensi**
Rahmada, A. ., & Susanto, E. R. (2025). Peningkatan Akurasi Prediksi Penyakit Jantung dengan Teknik SMOTEENN pada Algoritma Random Forest. _Jurnal Pendidikan Dan Teknologi Indonesia_, 4(12), 795-803. https://doi.org/10.52436/1.jpti.524

S. A. T. Al Azhima, D. Darmawan, N. F. Arief Hakim, I. Kustiawan, M. Al Qibtiya, dan N. S. Syafei, “Hybrid Machine Learning Model untuk memprediksi Penyakit Jantung dengan Metode Logistic Regression dan Random Forest”, j. teknologi terpadu, vol. 8, no. 1, hlm. 40–46, Jul 2022.

