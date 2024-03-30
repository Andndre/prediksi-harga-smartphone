import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import load_model, Sequential
import joblib

def predict_new_data():
    new_data = pd.DataFrame({
        'device': ['Apple'],
        'memory': [8],
        'internal_storage': [256],
        'dual_sim': [True],
        'esim': [True],
        'memory_slot': [True],
        '5g': [True],
        'nfc': [True],
        'body_length': [146.7],
        'body_width': [67.3],
        'body_thickness': [9.9],
        'battery': [5000],
        'display_type': ['IPS'],
        'display_size': [6.78],
        'display_res': [1080],
        'refresh_rate': [144],
        'display_hdr': [True],
    })
    
    # Load model, encoder dan normalizer
    model = joblib.load('model/my_model.keras')
    encoder = joblib.load('model/encoder.pkl')
    normalizer = joblib.load('model/normalizer.pkl')
    
    # Label Encoding
    new_data_encoded = new_data.copy()
    new_data_encoded['device'] = encoder.transform(new_data_encoded['device'])
    new_data_encoded['display_type'] = encoder.transform(new_data_encoded['display_type'])
    
    # Normalisasi
    new_data_normalized = normalizer.transform(new_data_encoded)
    
    print(new_data_normalized)
    
    
    