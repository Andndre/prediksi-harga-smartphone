# AI Prediksi Harga Smartphone
Tujuan dari project ini adalah menciptakan sebuah model AI yang dapat memprediksi harga smartphone berdasarkan berbagai fitur atau atribut yang dimiliki oleh smartphone tersebut. Project ini menggunakan metode regresi, yang memungkinkan untuk memprediksi nilai kontinu, dalam hal ini adalah harga dari smartphone.

## Apa itu Regresi?
Regresi adalah metode dalam machine learning yang digunakan untuk memprediksi nilai kontinu (angka). Berbeda dengan klasifikasi yang hanya menghasilkan output dalam bentuk kategori (misalnya prediksi cuaca: Hujan, Cerah, Panas).

## Algoritma yang Digunakan (Kandidat)
1. Linier Regresi
2. Ridge dan Lasso
3. Decision Tree
4. Random Forest
5. Gradient Boosting
6. Neural Network Regression

## Tahapan Pembuatan AI dengan Regresi
1. Cari Dataset
2. Bersihin Data (tidak boleh ada null atau kosong)
3. Label Encoding (ubah tipe data string numeric (int, float))
4. Normalisasi (meningkatkan kinerja model)
5. Pemisahan variabel target (y) dan fitur (X)
6. Pisahin dataframe menjadi 2 : data train (80% dari total), data test (20%)
7. Buat model (Algoritma)
8. Latih Model menggunakan data train
9. Testing dengan Data test
10. Cari persentase

## Persentase Keakuratan dari Setiap Algoritma
1. Linier Regression: 0.8146916750328905
2. Ridge Regression: 0.8542097305819878 (alpha 29.0)
3. Lasso Regression: 0.814836712339565 (alpha 1.0)
4. Decision Tree: 0.8483979773872512 (depth 6)
5. Random Forest: 0.9123759907926651 (n_estimators=99, max_depth=6, random_state=42)
6. Neural Network Regression: 0.9220560813939203
