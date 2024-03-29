import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# 1. Membaca dataset
data = pd.read_csv('dataset/Dataset Harga HP Indonesia.csv')

# 2. Memisahkan feature (X) dan target (y)
dropped_column = [ 'weight', 'os_price']
X = data.drop(dropped_column, axis=1) # fitur
y = data['os_price'] #target

# 3. Menggunakan LabelEncoder untuk mengkodekan setiap fitur non-numerik
encoder = LabelEncoder()
X_encoded = X.copy() # Membuat salinan untuk encoding fitur
X_encoded['device'] = encoder.fit_transform(X_encoded['device'])
X_encoded['display_type'] = encoder.fit_transform(X_encoded['display_type'])

# 4. Normalisasi fitur menggunakan StandardScaler
normalizer = StandardScaler() 
X_normalizer = normalizer.fit_transform(X_encoded)

# 5. Membagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X_normalizer, y, test_size=0.2, random_state=42)

# 6. Membuat Model dengan LinierRegression
model = LinearRegression()

# 7. Melatih model dengan menggunakan data train
model.fit(X_train, y_train)

# 8. Membuat prediksi menggunakan data test
y_pred = model.predict(X_test)

# 9. Menampilkan koefisien dan intersep dari model
print("Koefisien:", model.coef_)
print("Intersep:", model.intercept_)

# 10. Print MAE dan MSE
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('MAE : ', mae)
print('MSE : ', mse)
print('R2 Score : ', r2)

plt.figure(figsize=(8, 6))

# Plot nilai aktual (y_test) dengan warna merah
plt.scatter(np.arange(len(y_test)), y_test, color='red', label='Nilai Aktual')

# Plot nilai prediksi (y_pred) dengan warna biru
plt.scatter(np.arange(len(y_test)), y_pred, color='blue', label='Nilai Prediksi')

plt.xlabel('Index Data')
plt.ylabel('Nilai')
plt.title('Perbandingan Nilai Aktual dan Nilai Prediksi')
plt.legend()
plt.show()
