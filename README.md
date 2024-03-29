# AI Prediksi Harga Smartphone

## Regresi apa?
Prediksi yang outputnya berupa nilai kontinue (angka). Berbeda dengan Klasifikasi yang hanya bisa menghasilkan output berupa enum (misalnya dalam prediksi cuaca itu seperti Hujan, Cerah, Panas)

## Algoritma yang dipake (kandidat)
1. Linier Regresi
2. Ridge and Lasso
3. Decision Tree
4. Random Forest
5. Gradient Boosting 
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

## Persentase dari masing-masing algoritma
1. Linier Regression: 0.8146916750328905 
2. Ridge Regression: 0.8542097305819878 (alpha 29.0) 
3. Lasso Regression: 0.814836712339565 (alpha 1.0)
4. Decision Tree: 0.8483979773872512 (depth 6)
5. Random Forest: 0.9123759907926651 (n_estimators=99, max_depth=6, random_state=42)
6. Neural Network: 0.9177054408260952