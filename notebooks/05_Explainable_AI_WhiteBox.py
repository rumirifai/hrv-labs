"""
Fase 05: Explainable AI (XAI) - Ekstraksi Rumus Transparan White-Box
Deskripsi: Mengimplementasikan kerangka kerja Singh & Bura (2025) secara universal 
           untuk mentransformasikan model Deep Learning Black-Box (Keras) sensor tunggal
           menjadi rumusan persamaan linear murni yang dapat dinalar secara akademis.
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

from core import processors

# Menghilangkan log peringatan kompilasi biner TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Menghilangkan peringatan depresiasi fungsi dari Python/Pandas
warnings.filterwarnings("ignore")

# 1. Konfigurasi Jalur Direktori (Directory Paths)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "01_processed")
MODEL_DIR = os.path.join(BASE_DIR, "data", "03_saved_models")

def muat_dan_rekonstruksi_dataset_hrv():
    """
    Memuat dataset tabular HRV ECG hasil Fase 00, memisahkan fitur, 
    dan melakukan imputasi data kosong menggunakan teknik KNN Imputer.
    
    Returns:
    --------
    tuple (numpy.ndarray, numpy.ndarray, list)
        Sepasang array data terstandardisasi (X_scaled), label target (y),
        dan daftar nama fitur statistik.
    """
    print("LOG: [Memuat Dataset Fitur Statistik Terproses]")
    path_dataset = os.path.join(DATA_DIR, "Dataset_Fitur_ECG_6_Fitur.csv")
    
    if not os.path.exists(path_dataset):
        raise FileNotFoundError(f"Berkas dataset {path_dataset} tidak ditemukan. Jalankan Fase 00.")
        
    df = pd.read_csv(path_dataset)
    
    # Fokus pada 3 Fitur Utama Ranah Waktu sesuai riwayat eksperimen 
    fitur_cols = ["MeanNN_ms", "SDNN_ms", "RMSSD_ms"] 
    X_raw = df.filter(items=fitur_cols).values 
    
    # Transformasi string biner label target menjadi nilai integer numerik
    y = (df["Label"] == "Tinggi").astype(int).values 
    
    # Prosedur Imputasi Data Kosong dengan KNN Imputer (K=5) untuk penyelamatan data  
    print("LOG: Menerapkan algoritma KNN Imputer (N=5) untuk pembersihan data kontinu...")
    imputer = KNNImputer(n_neighbors=5)  
    X_imputed = imputer.fit_transform(X_raw)  
    
    # Standardisasi Fitur berbasis Skala Standard (Z-score)  
    scaler = StandardScaler()  
    X_scaled = scaler.fit_transform(X_imputed)  
    
    return X_scaled, y, fitur_cols

def evaluasi_dan_bedah_model_whitebox(X_scaled, y, fitur_cols):
    """
    Memuat berkas fisik model Black-Box terbaik, melakukan pengujian metrik metrologi lengkap,
    dan memanggil fungsi global White-Box extraction untuk menghasilkan persamaan matematika linear.
    """
    path_model = os.path.join(MODEL_DIR, "Model_cnn_lstm_ECG_ufr_Terbaik.keras")
    
    if not os.path.exists(path_model):
        # Jalankan fallback pencarian jika nama file model pasca-training fase 01 bervariasi
        path_model = os.path.join(MODEL_DIR, "Model_WhiteBox_KNN_Terbaik.keras")  
        
    if not os.path.exists(path_model):
        print(f"PERINGATAN: Berkas fisik model Black-Box tidak ditemukan pada direktori {MODEL_DIR}")
        print("            Pastikan Anda telah mengeksekusi Fase 01 dan menyimpan model terkait.")
        return

    print(f"\nLOG: Memuat arsitektur dan bobot parameter model Black-Box dari: {os.path.basename(path_model)}")
    model_ai = load_model(path_model) 
    
    # --- BAGIAN 1: EVALUASI METRIK PERFORMA KLASIFIKASI MODEL ---
    print("\nLOG: [Mengevaluasi Matriks Performa Klasifikasi]")
    
    # Reshape data ke dimensi 3D Keras temporal untuk kebutuhan inferensi deep neural network  
    X_3D = np.reshape(X_scaled, (len(X_scaled), len(fitur_cols), 1))  
    
    # Melakukan prediksi probabilitas kontinu kontinu  
    y_pred_prob = model_ai.predict(X_3D, verbose=0).flatten()  
    y_pred = (y_pred_prob > 0.5).astype(int) 
    
    # Mencetak Classification Report Komprehensif 
    print("\n" + "="*55)
    print("📊 LAPORAN METRIK KLASIFIKASI MODEL (DETEKSI UNIMODAL) 📊")
    print("="*55)
    print(classification_report(y, y_pred, target_names=["Beban Rendah (0)", "Beban Tinggi (1)"])) 
    print("="*55)
    
    # Mencetak Confusion Matrix secara terperinci dan aman dari kesalahan indeks terminal 
    cm = confusion_matrix(y, y_pred) 
    tn, fp, fn, tp = cm.ravel()
    print("\n🧩 INDEKS MATRIKS KEKACAUAN (CONFUSION MATRIX DETAILED) 🧩")
    print(f" -> True Negatives (TN)  : {tn} (Asli Rendah, Diprediksi Rendah)") 
    print(f" -> False Positives (FP) : {fp} (Asli Rendah, Diprediksi Tinggi [Alarm Palsu])") 
    print(f" -> False Negatives (FN) : {fn} (Asli Tinggi, Diprediksi Rendah [Kecolongan!])") 
    print(f" -> True Positives (TP)  : {tp} (Asli Tinggi, Diprediksi Tinggi)") 
    print("="*55)

    # --- BAGIAN 2: EKSTRAKSI PARAMETER WHITE-BOX UNIVERSAL ---
    # Memanggil framework global ekstraksi matematis yang berada pada core.processors 
    bobot_fitur, bias = processors.extract_whitebox_framework(
        model=model_ai,
        X_scaled=X_scaled,
        feature_names=fitur_cols,
        is_temporal=True
    )

def main():
    try:
        # 1. Mempersiapkan data pengujian ter-imputasi dan terstandardisasi 
        X_scaled, y, fitur_cols = muat_dan_rekonstruksi_dataset_hrv()
        
        # 2. Menjalankan framework dekonstruksi model Black-Box ke bentuk White-Box 
        evaluasi_dan_bedah_model_whitebox(X_scaled, y, fitur_cols)
        
    except Exception as e:
        print(f"ERROR: Terjadi kesalahan sistemis pada jalannya program. Pesan: {str(e)}")

if __name__ == "__main__":
    main()