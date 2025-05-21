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
1.Bagaimana cara mengembangkan model machine learning yang dapat memprediksi risiko penyakit jantung pada pasien berdasarkan data medis?
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


