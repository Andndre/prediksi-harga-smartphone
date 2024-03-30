import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import math

# 1. Membaca dataset
data = pd.read_csv('dataset/Dataset Harga HP Indonesia.csv')

data['os'] = np.where(data['device'].str.startswith('Apple'), 'IOS', 'Android')

# 2. Memisahkan feature (X) dan target (y)
dropped_column = ['device', 'weight', 'os_price']
X = data.drop(dropped_column, axis=1) # fitur
y = data['os_price'] #target

# 3. Menggunakan LabelEncoder untuk mengkodekan setiap fitur non-numerik
encoder = LabelEncoder()
X_encoded = X.copy() # Membuat salinan untuk encoding fitur
X_encoded['os'] = encoder.fit_transform(X_encoded['os'])
X_encoded['display_type'] = encoder.fit_transform(X_encoded['display_type'])

# 4. Normalisasi fitur menggunakan StandardScaler
normalizer = StandardScaler() 
X_normalizer = normalizer.fit_transform(X_encoded)

# 5. Membagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X_normalizer, y, test_size=0.2, random_state=42)

# 6. Membuat Model dengan Random Forest Regression
model = RandomForestRegressor(n_estimators=99, max_depth=6, random_state=42)

# 7. Melatih model dengan menggunakan data train
model.fit(X_train, y_train)

# 8. Membuat prediksi menggunakan data test
y_pred = model.predict(X_test)

print(X_test)

# 9. Menampilkan koefisien dan intersep dari model
# print("Koefisien:", model.coef_)
# print("Intersep:", model.intercept_)

# 10. Print MAE dan MSE
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('MAE : ', mae)
print('MSE : ', mse)
print('R2 Score : ', r2)

# plt.figure(figsize=(8, 6))

# # Plot nilai aktual (y_test) dengan warna merah
# plt.scatter(np.arange(len(y_test)), y_test, color='red', label='Nilai Aktual')

# # Plot nilai prediksi (y_pred) dengan warna biru
# plt.scatter(np.arange(len(y_test)), y_pred, color='blue', label='Nilai Prediksi')

# plt.xlabel('Index Data')
# plt.ylabel('Nilai')
# plt.title('Perbandingan Nilai Aktual dan Nilai Prediksi')
# plt.legend()
# plt.show()

def predict_new_data():
    new_data = pd.DataFrame({
        'memory': [8],
        'internal_storage': [256],
        'dual_sim': [False],
        'esim': [False],
        'memory_slot': [True],
        '5g': [False],
        'nfc': [False],
        'body_length': [146.7],
        'body_width': [67.3],
        'body_thickness': [9.9],
        'battery': [5000],
        'display_type': ['IPS'],
        'display_size': [6.78],
        'display_res': [720],
        'refesh_rate': [144],
        'display_hdr': [False],
        'os': ['Android']
    })
    
    # Label Encoding
    new_data_encoded = new_data.copy()
    new_data_encoded['os'] = encoder.fit_transform(new_data_encoded['os'])
    new_data_encoded['display_type'] = encoder.fit_transform(new_data_encoded['display_type'])
    
    # Normalisasi
    new_data_normalized = normalizer.transform(new_data_encoded)
    
    prediksi = model.predict(new_data_normalized)
    print('Rp. ' + str(math.ceil(prediksi[0]) * 1000))


predict_new_data()
