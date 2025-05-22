# **Laporan Proyek Machine Learning Terapan | Prediksi Penyakit Jantung**"
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

## Referensi
Rahmada, A. ., & Susanto, E. R. (2025). Peningkatan Akurasi Prediksi Penyakit Jantung dengan Teknik SMOTEENN pada Algoritma Random Forest. _Jurnal Pendidikan Dan Teknologi Indonesia_, 4(12), 795-803. https://doi.org/10.52436/1.jpti.524

S. A. T. Al Azhima, D. Darmawan, N. F. Arief Hakim, I. Kustiawan, M. Al Qibtiya, dan N. S. Syafei, “Hybrid Machine Learning Model untuk memprediksi Penyakit Jantung dengan Metode Logistic Regression dan Random Forest”, j. teknologi terpadu, vol. 8, no. 1, hlm. 40–46, Jul 2022.


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





