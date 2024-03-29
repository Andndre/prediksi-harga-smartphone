# AI Prediksi Harga Mobil

## Regresi apa?
Prediksi yang outputnya berupa nilai kontinue (angka). Berbeda dengan Klasifikasi yang hanya bisa menghasilkan output berupa enum (misalnya dalam prediksi cuaca itu seperti Hujan, Cerah, Panas)

## Algoritma yang dipake (kandidat)
1. Linier Regresi
2. Ridge and Lasso
3. Decision Tree
4. Random Forest
5. Gradient Boosting
6. Support Vector Regression 
7. Neural Network Regression

## Tahapan membuat AI dengan Regresi
1. Cari Dataset
2. Bersihin Data (Ga boleh ada null atau kosong)
3. Label Encoding (ubah tipe data string numeric (int, float))
4. Normalisasi (Intinya meningkatkan kinerja model)
5. Pemisahan variabel target (y) dan fitur (X)
6. Pisahin dataframe menjadi 2 : data train (80% dari total), data test (20%)
7. Bikin model (Algoritma)
8. Latih Model menggunakan data train
9. Coba testing dengan Data test
10. Cari persentase 
