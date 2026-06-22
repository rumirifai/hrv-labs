"""
Fase 01: Pemodelan Berbasis Fitur Statistik Tabular (Unimodal Sensor)
Deskripsi: Melatih dan mengevaluasi performa model Machine Learning klasik (SVM, RF) 
           and Deep Learning (CNN-LSTM, LSTM-MHA) menggunakan 10-Fold Cross Validation.
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import class_weight

from core import processors, models

# Menghilangkan log peringatan kompilasi biner TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Menghilangkan peringatan depresiasi fungsi dari Python/Pandas
warnings.filterwarnings("ignore")

# Konfigurasi Jalur Direktori
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "01_processed")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "data", "03_saved_models")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

def evaluasi_machine_learning_klasik(df, nama_sensor, metode="svm"):
    """Mengevaluasi model SVM atau Random Forest menggunakan 10-Fold Cross Validation."""
    fitur_cols = ["MeanNN_ms", "SDNN_ms", "RMSSD_ms"]
    
    # Sinkronisasi baris data kosong (dropna secara serentak antara X dan y)
    df_clean = df.dropna(subset=fitur_cols)
    X = df_clean[fitur_cols].values
    y_text = df_clean["Label"].values.flatten()
    
    print(f"LOG: Dimensi data tersinkronisasi -> X: {X.shape}, y: {y_text.shape}")
    
    le = LabelEncoder()
    y = le.fit_transform(y_text)
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    acc_l, prec_l, rec_l, f1_l = [], [], [], []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = np.take(X, train_idx, axis=0), np.take(X, test_idx, axis=0)
        y_train, y_test = np.take(y, train_idx, axis=0), np.take(y, test_idx, axis=0)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if metode == "svm":
            model = SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
            
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        acc_l.append(accuracy_score(y_test, y_pred))
        prec_l.append(precision_score(y_test, y_pred, zero_division=0))
        rec_l.append(recall_score(y_test, y_pred, zero_division=0))
        f1_l.append(f1_score(y_test, y_pred, zero_division=0))
        
    processors.print_kfold_evaluation_summary(acc_l, prec_l, rec_l, f1_l, f"{metode.upper()} - {nama_sensor}")

def evaluasi_deep_learning_unimodal(df, nama_sensor, arsitektur="cnn_lstm", seleksi_fitur=None):
    """Mengevaluasi arsitektur CNN-LSTM atau LSTM-Attention dengan opsi pembobotan UFR/MRMR."""
    fitur_cols = ["MeanNN_ms", "SDNN_ms", "RMSSD_ms"]
    
    # Sinkronisasi baris data kosong agar seimbang sebelum pembagian k-fold
    df_clean = df.dropna(subset=fitur_cols)
    X = df_clean[fitur_cols].values
    y_text = df_clean["Label"].values.flatten()
    
    le = LabelEncoder()
    y = le.fit_transform(y_text)
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    acc_l, prec_l, rec_l, f1_l = [], [], [], []
    best_acc = 0.0
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = np.take(X, train_idx, axis=0), np.take(X, test_idx, axis=0)
        y_train, y_test = np.take(y, train_idx, axis=0), np.take(y, test_idx, axis=0)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Prosedur Seleksi dan Pembobotan Fitur (Feature Weighting)
        if seleksi_fitur == "ufr":
            weights = processors.compute_ufr_weights(X_train_scaled, y_train)
            X_train_scaled = np.multiply(X_train_scaled, weights)
            X_test_scaled = np.multiply(X_test_scaled, weights)
        elif seleksi_fitur == "mrmr":
            weights = processors.compute_mrmr_weights(X_train_scaled, y_train, fitur_cols)
            X_train_scaled = np.multiply(X_train_scaled, weights)
            X_test_scaled = np.multiply(X_test_scaled, weights)
            
        # Reshaping ke bentuk tensor 3D Keras (batch, timestep, fitur)
        X_train_reshaped = X_train_scaled.reshape((len(X_train_scaled), len(fitur_cols), 1))
        X_test_reshaped = X_test_scaled.reshape((len(X_test_scaled), len(fitur_cols), 1))
        
        # Membangun Model dari core.models
        if arsitektur == "cnn_lstm":
            model = models.build_cnn_lstm_tabular(input_dim=len(fitur_cols))
        else:
            model = models.build_lstm_attention_network(input_dim=len(fitur_cols))
            
        # Penanganan Ketidakseimbangan Data lokal per Fold
        cls_w = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        dict_cls_w = {0: cls_w.item(0), 1: cls_w.item(1)}
        
        model.fit(X_train_reshaped, y_train, epochs=50, batch_size=16, verbose=0, class_weight=dict_cls_w)
        
        y_pred = (model.predict(X_test_reshaped, verbose=0) > 0.5).astype(int).flatten()
        acc = accuracy_score(y_test, y_pred)
        
        acc_l.append(acc)
        prec_l.append(precision_score(y_test, y_pred, zero_division=0))
        rec_l.append(recall_score(y_test, y_pred, zero_division=0))
        f1_l.append(f1_score(y_test, y_pred, zero_division=0))
        
        # Simpan checkpoint model fisik terbaik
        if acc > best_acc:
            best_acc = acc
            suffix = f"_{seleksi_fitur}" if seleksi_fitur else ""
            model_path = os.path.join(MODEL_SAVE_DIR, f"Model_{arsitektur}_{nama_sensor}{suffix}_Terbaik.keras")
            model.save(model_path)
            
    identifikasi = f"{arsitektur.upper()}" + (f" + {seleksi_fitur.upper()}" if seleksi_fitur else "")
    processors.print_kfold_evaluation_summary(acc_l, prec_l, rec_l, f1_l, f"{identifikasi} - {nama_sensor}")

def main():
    # Memuat file hasil pemrosesan fase 00
    path_ecg = os.path.join(DATA_DIR, "Dataset_Fitur_ECG_6_Fitur.csv")
    path_ppg = os.path.join(DATA_DIR, "Dataset_Fitur_PPG_6_Fitur.csv")
    
    if not (os.path.exists(path_ecg) and os.path.exists(path_ppg)):
        print("ERROR: File CSV dataset terproses tidak ditemukan. Jalankan file 00 terlebih dahulu.")
        return
        
    df_ecg = pd.read_csv(path_ecg)
    df_ppg = pd.read_csv(path_ppg)
    
    # Menampilkan informasi keseimbangan kelas target
    processors.inspect_data_balance(df_ecg['Label'].values)
    
    # 1. Eksperimen Machine Learning Klasik (Folder 4 & Folder 7)
    print("\nLOG: [Eksperimen 1: Algoritma Machine Learning Klasik]")
    evaluasi_machine_learning_klasik(df_ecg, "ECG", metode="svm")
    evaluasi_machine_learning_klasik(df_ppg, "PPG", metode="svm")
    evaluasi_machine_learning_klasik(df_ecg, "ECG", metode="rf")
    evaluasi_machine_learning_klasik(df_ppg, "PPG", metode="rf")
    
    # 2. Eksperimen Deep Learning Baseline (Folder 2 & Folder 3)
    print("\nLOG: [Eksperimen 2: Algoritma Deep Learning Baseline]")
    evaluasi_deep_learning_unimodal(df_ecg, "ECG", arsitektur="cnn_lstm")
    evaluasi_deep_learning_unimodal(df_ppg, "PPG", arsitektur="cnn_lstm")
    
    # 3. Eksperimen Seleksi Fitur UFR & MRMR (Folder 5 & Folder 6)
    print("\nLOG: [Eksperimen 3: Optimasi Seleksi Fitur (UFR & MRMR)]")
    evaluasi_deep_learning_unimodal(df_ecg, "ECG", arsitektur="cnn_lstm", seleksi_fitur="ufr")
    evaluasi_deep_learning_unimodal(df_ppg, "PPG", arsitektur="cnn_lstm", seleksi_fitur="ufr")
    evaluasi_deep_learning_unimodal(df_ecg, "ECG", arsitektur="cnn_lstm", seleksi_fitur="mrmr")
    evaluasi_deep_learning_unimodal(df_ppg, "PPG", arsitektur="cnn_lstm", seleksi_fitur="mrmr")
    
    # 4. Eksperimen LSTM + Multi-Head Attention Mechanism (Folder 8)
    print("\nLOG: [Eksperimen 4: Jaringan Saraf LSTM + Multi-Head Attention]")
    evaluasi_deep_learning_unimodal(df_ecg, "ECG", arsitektur="lstm_attention")
    evaluasi_deep_learning_unimodal(df_ppg, "PPG", arsitektur="lstm_attention")

if __name__ == "__main__":
    main()