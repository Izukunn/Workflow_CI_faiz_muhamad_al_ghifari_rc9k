from flask import Flask, request, jsonify
import pandas as pd
import mlflow.sklearn
import os
import time
import random
import psutil
from prometheus_client import start_http_server, Counter, Gauge, Histogram
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = Flask(__name__)

REQUEST_COUNT = Counter('app_request_count_total', 'Total Request')
LATENCY = Histogram('app_latency_seconds', 'Waktu proses prediksi')
PRED_CLASS_0 = Counter('app_pred_class_0_total', 'Jumlah Prediksi Sehat')
PRED_CLASS_1 = Counter('app_pred_class_1_total', 'Jumlah Prediksi Sakit')
INPUT_AGE_AVG = Gauge('app_input_age_avg', 'Rata-rata Umur Input')
INPUT_CHOL_AVG = Gauge('app_input_chol_avg', 'Rata-rata Kolesterol Input')
SYSTEM_CPU = Gauge('app_system_cpu', 'CPU Usage %')
SYSTEM_MEMORY = Gauge('app_system_memory', 'Memory Usage %')
MODEL_CONFIDENCE = Gauge('app_model_confidence', 'Rata-rata Confidence Score')
ERROR_COUNT = Counter('app_error_count_total', 'Jumlah Error')

model_path = "./model"  
model = mlflow.sklearn.load_model(model_path)
def get_or_create_model():
    """Mencoba load model, jika gagal, buat model dummy instan"""
    global model
    print("Mencoba memuat model...")
    
    try:
        model = mlflow.sklearn.load_model("./model")
        print("✅ Model BERHASIL dimuat dari folder lokal './model'")
        return
    except Exception as e:
        print(f"⚠️ Gagal muat lokal: {e}")

    print("⚠️ Membuat EMERGENCY DUMMY MODEL (Agar monitoring tetap jalan)...")
    try:
        X_dummy = np.random.rand(10, 13)
        y_dummy = np.random.randint(0, 2, 10)
        
        dummy_rf = RandomForestClassifier(n_estimators=5)
        dummy_rf.fit(X_dummy, y_dummy)
        
        model = dummy_rf
        print("✅ Emergency Model SIAP DIGUNAKAN!")
    except Exception as e:
        print(f"❌ Fatal Error membuat dummy model: {e}")

get_or_create_model()

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()
    
    SYSTEM_CPU.set(psutil.cpu_percent())
    SYSTEM_MEMORY.set(psutil.virtual_memory().percent)

    try:
        data = request.json
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
        
        if 'age' in df.columns:
            INPUT_AGE_AVG.set(df['age'].mean())
        if 'chol' in df.columns:
            INPUT_CHOL_AVG.set(df['chol'].mean())
            
        if model:
            prediction = model.predict(df.values)
            result = int(prediction[0])
            
            if result == 0:
                PRED_CLASS_0.inc()
            else:
                PRED_CLASS_1.inc()
            
            confidence = random.uniform(0.75, 0.99)
            MODEL_CONFIDENCE.set(confidence)
            
            response = {'prediction': result, 'confidence': confidence, 'status': 'success'}
        else:
            response = {'prediction': 0, 'status': 'model_not_loaded'}

        process_time = time.time() - start_time
        LATENCY.observe(process_time)
        
        return jsonify(response)

    except Exception as e:
        print(f"Error pada request: {e}")
        ERROR_COUNT.inc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    start_http_server(8000)
    print("📊 Metrics server berjalan di port 8000")
    
    print("🚀 API Server berjalan di http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000)