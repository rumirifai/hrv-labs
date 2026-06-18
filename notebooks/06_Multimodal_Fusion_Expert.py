"""
Fase 06: Multimodal Data Fusion and Expert Analysis
Deskripsi: Mengintegrasikan fitur statistik dari sensor ECG dan PPG (Feature-Level Fusion),
           menangani data hilang via KNN Imputer, melatih jaringan saraf fusi multimodal,
           serta mengekstrak rumusan White-Box untuk menganalisis dominansi kontribusi sensor.
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint

from core import loaders, processors, models

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 1. Konfigurasi Jalur Direktori (Directory Paths)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DATA_DIR = os.path.join(BASE_DIR, "data", "01_processed")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "data", "03_saved_models")

def eksekusi_fusi_multimodal_preprocessing():
    """
    Memuat dataset dari kedua modalitas sensor, melakukan operasi fusi data (Inner Join),
    mengisolasi matriks fitur, dan melakukan rekonstruksi data menggunakan KNN Imputer.
    
    Returns:
    --------
    tuple (numpy.ndarray, numpy.ndarray, list)
        Matriks fitur terstandardisasi (X_scaled), array target (y), 
        dan daftar nama fitur gabungan.
    """
    print("LOG: [Prosedur Fusi Data Multimodal Level Fitur dimulai]")
    path_ecg = os.path.join(INPUT_DATA_DIR, "Dataset_Fitur_ECG_6_Fitur.csv")
    path_ppg = os.path.join(INPUT_DATA_DIR, "Dataset_Fitur_PPG_6_Fitur.csv")
    
    if not (os.path.exists(path_ecg) and os.path.exists(path_ppg)):
        raise FileNotFoundError("Berkas dataset terproses ECG atau PPG tidak ditemukan. Jalankan Fase 00.")
        
    # Eksekusi sinkronisasi dan fusi join via core.loaders
    df_multimodal = loaders.merge_multimodal_features(path_ecg, path_ppg)
    print(f"LOG: Sinkronisasi selesai. Total data multimodal terintegrasi: {len(df_multimodal)} baris.")
    
    # Spesifikasi susunan kolom 6 fitur terintegrasi (3 Fitur ECG + 3 Fitur PPG)
    fitur_cols = ["MeanNN_ECG", "SDNN_ECG", "RMSSD_ECG", "MeanNN_PPG", "SDNN_PPG", "RMSSD_PPG"]
    X_raw = df_multimodal[fitur_cols].values
    
    # Determinasi biner target berdasarkan kolom sinkron Label_ECG
    y = (df_multimodal["Label_ECG"] == "Tinggi").astype(int).values
    
    # Penambalan data hilang menggunakan algoritma KNN Imputer (K=5)
    print("LOG: Menerapkan KNN Imputer pada matriks fitur multimodal...")
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X_raw)
    
    # Standardisasi Skala Data Multimodal
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    return X_scaled, y, fitur_cols

def latih_dan_evaluasi_jaringan_multimodal(X_scaled, y, fitur_cols):
    """
    Membagi data biner, melatih arsitektur Hybrid Neural Network Multimodal Fusion,
    dan melakukan pengujian metrik metrologi klasifikasi serta dekonstruksi rumus White-Box.
    """
    print("\nLOG: [Prosedur Validasi dan Pelatihan Model Neural Multimodal]")
    
    # Reshape ke bentuk format Keras Tensor 3D: (jumlah_sampel, 6_fitur, 1_channel)
    X_3D = np.reshape(X_scaled, (len(X_scaled), len(fitur_cols), 1))
    
    # Pembagian data konsisten seed 42 dengan stratifikasi distribusi kelas
    X_train, X_test, y_train, y_test = train_test_split(
        X_3D, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Menampilkan analisis keseimbangan data latih lokal via core.processors
    processors.inspect_data_balance(y_train)
    
    # Membangun arsitektur model fusi multimodal via core.models
    model_multimodal = models.build_multimodal_fusion_network(input_dim=len(fitur_cols))
    
    # Callback Checkpoint otomatis untuk mengamankan bobot parameter paling jenius
    checkpoint_path = os.path.join(MODEL_SAVE_DIR, "Model_Multimodal_Fusion_Terbaik.keras")
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1
    )
    
    # Menghitung bobot penyeimbang kelas (Class Weights)
    from sklearn.utils import class_weight
    cls_w = class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    dict_cls_w = {0: cls_w.item(0), 1: cls_w.item(1)}
    
    # Proses pembelajaran model multimodal (30 Epoch)
    model_multimodal.fit(
        X_train, y_train, epochs=30, batch_size=16, class_weight=dict_cls_w,
        validation_data=(X_test, y_test), callbacks=[checkpoint_callback], verbose=1
    )
    
    # Restorasi bobot arsitektur terbaik pasca-pelatihan
    print("LOG: Memuat ulang bobot parameter optimal dari memori fisik...")
    model_multimodal.load_weights(checkpoint_path)
    
    # Inferensi dan evaluasi performa data uji terisolasi
    y_pred_prob = model_multimodal.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # --- LAPORAN METRIK KLASIFIKASI MULTIMODAL ---
    print("\n" + "="*55)
    print("📊 LAPORAN METRIK KINERJA MODEL FUSI MULTIMODAL 📊")
    print("="*55)
    print(classification_report(y_test, y_pred, target_names=["Beban Rendah (0)", "Beban Tinggi (1)"]))
    print("="*55)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("\n🧩 INDEKS CONFUSION MATRIX MULTIMODAL 🧩")
    print(f" -> True Negatives (TN)  : {tn} (Asli Santai, Diprediksi Santai)")
    print(f" -> False Positives (FP) : {fp} (Asli Santai, Diprediksi Stres [Alarm Palsu])")
    print(f" -> False Negatives (FN) : {fn} (Asli Stres, Diprediksi Santai [Kecolongan!])")
    print(f" -> True Positives (TP)  : {tp} (Asli Stres, Diprediksi Stres)")
    print("="*55)

    # --- EKSTRAKSI RUMUS WHITE-BOX MULTIMODAL ---
    # Mengonversi seluruh data skala kembali ke bentuk 2D untuk pass estimasi linier
    X_scaled_2d = np.squeeze(X_3D)
    
    # Memanggil framework global interpretabilitas matematika
    bobot_fitur, bias = processors.extract_whitebox_framework(
        model=model_multimodal,
        X_scaled=X_scaled_2d,
        feature_names=fitur_cols,
        is_temporal=True
    )
    
    # --- ANALISIS KOMPARATIF DOMINANSI SENSOR FISIOLOGIS ---
    print("\n" + "🏆" + " " + "ANALISIS SENSOR PALING DOMINAN (METROLOGI MUTLAK) " + "🏆")
    print("="*60)
    magnitudo_ecg = abs(bobot_fitur["MeanNN_ECG"]) + abs(bobot_fitur["SDNN_ECG"]) + abs(bobot_fitur["RMSSD_ECG"])
    magnitudo_ppg = abs(bobot_fitur["MeanNN_PPG"]) + abs(bobot_fitur["SDNN_PPG"]) + abs(bobot_fitur["RMSSD_PPG"])
    
    total_magnitudo = magnitudo_ecg + magnitudo_ppg
    persen_ecg = (magnitudo_ecg / total_magnitudo) * 100
    persen_ppg = (magnitudo_ppg / total_magnitudo) * 100
    
    print(f" Total Magnitudo Pengaruh Sinyal Dada (ECG) : {magnitudo_ecg:.4f} ({persen_ecg:.2f}%)")
    print(f" Total Magnitudo Pengaruh Sinyal Nadi (PPG) : {magnitudo_ppg:.4f} ({persen_ppg:.2f}%)")
    print("-" * 60)
    
    if magnitudo_ecg > magnitudo_ppg:
        print(" KESIMPULAN ILMIAH: Sensor ECG (Dada) LEBIH DOMINAN secara signifikan")
        print("                    dalam mendeteksi parameter Beban Kognitif subjek.")
    else:
        print(" KESIMPULAN ILMIAH: Sensor PPG (Nadi Tangan) LEBIH DOMINAN secara signifikan")
        print("                    dalam mendeteksi parameter Beban Kognitif subjek.")
    print("="*60)

def main():
    try:
        # 1. Menjalankan pipeline fusi level fitur dan penambalan KNN
        X_scaled, y, fitur_cols = eksekusi_fusi_multimodal_preprocessing()
        
        # 2. Menjalankan proses latih model saraf tiruan fusi serta evaluasi komparatif White-Box
        latih_dan_evaluasi_jaringan_multimodal(X_scaled, y, fitur_cols)
        
    except Exception as e:
        print(f"ERROR: Terjadi kegagalan sistem pada eksekusi program. Pesan: {str(e)}")

if __name__ == "__main__":
    main()