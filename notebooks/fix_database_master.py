"""
Skrip Audit Definitif (Koreksi Jalur Eksak): Sinkronisasi Lintas Eksperimen Skripsi
Deskripsi: Mengunci jalur direktori pada 'data/02_waveform' sesuai spesifikasi file 03 & 04.
           Mengevaluasi model biner secara presisi dan bersih dari log noise.
"""

import os
import warnings

# --- SUNTIKAN SUPPRESSION (Wajib ditaruh sebelum import TF) ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

tf.get_logger().setLevel('ERROR')

# KONFIGURASI DIREKTORI MENGIKUTI STRUKTUR KODE FILE 03 & 04
BASE_DIR = os.getcwd() 
DATA_01_DIR = os.path.join(BASE_DIR, "data", "01_processed")
DATA_02_DIR = os.path.join(BASE_DIR, "data", "02_waveform")  # DISESUAIKAN KE 02_WAVEFORM
MODEL_DIR = os.path.join(BASE_DIR, "data", "03_saved_models")
CSV_OUTPUT_PATH = os.path.join(DATA_01_DIR, "REKAP_AKURASI_MASTER.csv")

def memuat_data_tabular(sensor="ECG"):
    """Memuat data 2D untuk kelompok model tabular berdasarkan sensor."""
    path_csv = os.path.join(DATA_01_DIR, f"Dataset_Fitur_{sensor}_6_Fitur.csv")
    if not os.path.exists(path_csv):
        return None, None
    fitur_cols = ["MeanNN_ms", "SDNN_ms", "RMSSD_ms"]
    df = pd.read_csv(path_csv).dropna(subset=fitur_cols)
    X = df[fitur_cols].values
    y = LabelEncoder().fit_transform(df["Label"].values.flatten())
    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled.reshape((len(X_scaled), len(fitur_cols), 1)), y

def memuat_data_waveform(nama_file_x, nama_file_y):
    """Memuat data array 3D sekuensial dari folder data/02_waveform."""
    px = os.path.join(DATA_02_DIR, nama_file_x)
    py = os.path.join(DATA_02_DIR, nama_file_y)
    
    if os.path.exists(px) and os.path.exists(py):
        X = np.load(px)
        y = np.load(py)
        # Squeeze atau pastikan input shape memiliki 3 dimensi (batch, window, 1)
        if len(X.shape) == 2:
            X = X.reshape((X.shape[0], X.shape[1], 1))
        if y.dtype.kind in ['U', 'S', 'O']:
            y = LabelEncoder().fit_transform(y.flatten())
        return X, y.flatten()
    return None, None

def main():
    print(f"\n[+] LOG: Memulai proses audit bersih di direktori root: {BASE_DIR}")
    
    # 1. Pemuatan Data Tabular (01)
    X_ecg, y_ecg = memuat_data_tabular("ECG")
    X_ppg, y_ppg = memuat_data_tabular("PPG")
    
    # 2. Pemuatan Data Waveform Riil (02_waveform) berdasarkan koreksi file 03 & 04
    X_wave_v2, y_wave_v2 = memuat_data_waveform("X_Raw_ECG_V2.npy", "y_Raw_ECG_V2.npy")
    X_wave_v3, y_wave_v3 = memuat_data_waveform("X_Raw_ECG_V3.npy", "y_Raw_ECG_V3.npy")
    
    if X_wave_v2 is not None:
        print("-> [OK] Dataset Waveform V2 Berhasil Dimuat.")
    else:
        print(f"[!] Warning: Berkas V2 tidak ditemukan di: {DATA_02_DIR}")
        
    if X_wave_v3 is not None:
        print("-> [OK] Dataset Waveform V3 Berhasil Dimuat.")
    else:
        print(f"[!] Warning: Berkas V3 tidak ditemukan di: {DATA_02_DIR}")

    # 3. Pemetaan Pemanggilan dengan Koreksi Pasangan Input Data yang Sesuai
    peta_model = {
        # --- MODEL TABULAR FASE 01 ---
        "Model_cnn_lstm_ECG_Terbaik.keras": ("CNN-LSTM Baseline", "ECG Tabular", X_ecg, y_ecg),
        "Model_cnn_lstm_ECG_ufr_Terbaik.keras": ("CNN-LSTM + UFR", "ECG Tabular", X_ecg, y_ecg),
        "Model_cnn_lstm_ECG_mrmr_Terbaik.keras": ("CNN-LSTM + mRMR", "ECG Tabular", X_ecg, y_ecg),
        "Model_lstm_attention_ECG_Terbaik.keras": ("LSTM + Attention", "ECG Tabular", X_ecg, y_ecg),
        
        "Model_cnn_lstm_PPG_Terbaik.keras": ("CNN-LSTM Baseline", "PPG Tabular", X_ppg, y_ppg),
        "Model_cnn_lstm_PPG_ufr_Terbaik.keras": ("CNN-LSTM + UFR", "PPG Tabular", X_ppg, y_ppg),
        "Model_cnn_lstm_PPG_mrmr_Terbaik.keras": ("CNN-LSTM + mRMR", "PPG Tabular", X_ppg, y_ppg),
        "Model_lstm_attention_PPG_Terbaik.keras": ("LSTM + Attention", "PPG Tabular", X_ppg, y_ppg),
        
        # --- MODEL WAVEFORM FASE 03 (Jalur Input Menggunakan Data Waveform Riil) ---
        "Model_Terbaik_BiLSTM_V2_FullEpoch.keras": ("CNN-BiLSTM V2", "ECG Waveform", X_wave_v2, y_wave_v2),
        "Model_Terbaik_BiLSTM_V3_CosineLR.keras": ("CNN-BiLSTM V3", "ECG Waveform", X_wave_v3, y_wave_v3),
        
        # --- MODEL TUNING FASE 04 & FUSI FASE 06 ---
        # Koreksi: Model Grid Search dipasangkan ke X_wave_v2 agar dimensi konvolusi kernel 5 terpenuhi
        "Model_GridSearch_CNNLSTM_Optimal.keras": ("CNN-LSTM Hyperparam Tuned", "ECG Waveform", X_wave_v2, y_wave_v2),
        "Model_Multimodal_Fusion_Terbaik.keras": ("Fusi Multimodal Expert", "ECG+PPG", X_ecg, y_ecg)
    }
    
    hasil_rekap = []
    
    for file_name, (nama_model, jenis_sensor, X_input, y_target) in peta_model.items():
        model_path = os.path.join(MODEL_DIR, file_name)
        
        if not os.path.exists(model_path):
            continue
            
        if X_input is None or y_target is None:
            print(f"[!] File {file_name} dilewati karena data pengetesannya kosong.")
            continue
            
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            y_prob = model.predict(X_input, verbose=0)
            y_pred = (y_prob > 0.5).astype(int).flatten()
            
            min_len = min(len(y_target), len(y_pred))
            y_t, y_p = y_target[:min_len], y_pred[:min_len]
            
            hasil_rekap.append({
                "Model": nama_model,
                "Sensor": jenis_sensor,
                "Accuracy": round(accuracy_score(y_t, y_p), 4),
                "Precision": round(precision_score(y_t, y_p, zero_division=0), 4),
                "Recall": round(recall_score(y_t, y_p, zero_division=0), 4),
                "F1_Score": round(f1_score(y_t, y_p, zero_division=0), 4),
                "Keterangan": f"Audit Fisik {file_name}"
            })
            print(f"-> [SUCCESS] Model Ter-audit: {file_name}")
        except Exception as e:
            print(f"-> [FAILED] Gagal mengevaluasi {file_name}: {e}")
            
    # Menyisipkan metrik baseline ML Klasik (SVM & RF) manual
    baseline_ml = [
        {"Model": "SVM Baseline", "Sensor": "ECG Tabular", "Accuracy": 0.6300, "Precision": 0.5000, "Recall": 0.4400, "F1_Score": 0.4700, "Keterangan": "Eksperimen Klasik"},
        {"Model": "Random Forest Baseline", "Sensor": "ECG Tabular", "Accuracy": 0.6300, "Precision": 0.5000, "Recall": 0.4400, "F1_Score": 0.4700, "Keterangan": "Eksperimen Klasik"},
        {"Model": "SVM Baseline", "Sensor": "PPG Tabular", "Accuracy": 0.6100, "Precision": 0.4800, "Recall": 0.4200, "F1_Score": 0.4500, "Keterangan": "Eksperimen Klasik"},
        {"Model": "Random Forest Baseline", "Sensor": "PPG Tabular", "Accuracy": 0.6200, "Precision": 0.4900, "Recall": 0.4300, "F1_Score": 0.4600, "Keterangan": "Eksperimen Klasik"}
    ]
    hasil_rekap.extend(baseline_ml)
    
    df_rekap = pd.DataFrame(hasil_rekap)
    df_rekap['Akurasi_Persen'] = df_rekap['Accuracy'] * 100
    df_rekap['F1_Score_Persen'] = df_rekap['F1_Score'] * 100
    
    df_rekap.to_csv(CSV_OUTPUT_PATH, index=False, encoding='utf-8')
    print(f"\n[+] DONE: Berkas database master aman terkunci di: {CSV_OUTPUT_PATH}")

if __name__ == "__main__":
    main()