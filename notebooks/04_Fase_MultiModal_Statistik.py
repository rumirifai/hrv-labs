"""
Fase 04: Analisis Baseline Model dan Optimasi Hyperparameter (Grid Search)
Deskripsi: Memvalidasi proporsi ketidakseimbangan data target, melatih model baseline 
           Logistic Regression, serta menjalankan prosedur Grid Search otomatis untuk 
           mencari kombinasi parameter CNN-LSTM yang paling optimal pada dataset V2.
"""

import os
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from core import processors, models

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 1. Konfigurasi Jalur Direktori (Directory Paths)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "02_waveform")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "data", "03_saved_models")

def analisis_dan_pelatihan_baseline_logreg():
    """
    Memuat dataset waveform V2, menganalisis rasio ketidakseimbangan kelas target, 
    dan melatih model baseline Logistic Regression menggunakan representasi data 2D.
    """
    print("\nLOG: [Tahap 1: Analisis Distribusi Data & Evaluasi Baseline Model]")
    
    path_x = os.path.join(DATA_DIR, "X_Raw_ECG_V2.npy")
    path_y = os.path.join(DATA_DIR, "y_Raw_ECG_V2.npy")
    
    if not (os.path.exists(path_x) and os.path.exists(path_y)):
        print("ERROR: Berkas dataset V2 (.npy) tidak ditemukan. Jalankan Fase 02 terlebih dahulu.")
        return
        
    X_3D = np.load(path_x)
    y = np.load(path_y)
    
    # Menampilkan kalkulasi rasio distribusi kelas target via core.processors
    processors.inspect_data_balance(y) 
    
    # Transformasi dimensi tensor 3D menjadi matriks 2D menggunakan np.squeeze untuk model linear
    X_2D = np.squeeze(X_3D) 
    print(f"LOG: Dimensi matriks pasca-transformasi (Squeeze): {X_2D.shape}") 
    
    # Pembagian dataset dengan proporsi 80% Latih dan 20% Uji
    X_train, X_test, y_train, y_test = train_test_split(
        X_2D, y, test_size=0.2, random_state=42, stratify=y
    ) 
    
    print("LOG: Melatih model baseline Logistic Regression...")
    model_lr = LogisticRegression(max_iter=1000, random_state=42) 
    model_lr.fit(X_train, y_train) 
    
    y_pred = model_lr.predict(X_test) 
    
    print("\n" + "-" * 50)
    print("--- HASIL EVALUASI BASELINE LOGISTIC REGRESSION ---")
    print("-" * 50)
    print(f" -> Akurasi   : {accuracy_score(y_test, y_pred) * 100:.2f}%") 
    print(f" -> Precision : {precision_score(y_test, y_pred, zero_division=0) * 100:.2f}%") 
    print(f" -> Recall    : {recall_score(y_test, y_pred, zero_division=0) * 100:.2f}%") 
    print(f" -> F1-Score  : {f1_score(y_test, y_pred, zero_division=0) * 100:.2f}%") 
    print("-" * 50)

def eksekusi_grid_search_cnn_lstm():
    """
    Menjalankan simulasi Hyperparameter Tuning (Grid Search) otomatis secara murni 
    untuk menemukan kombinasi optimal pada komponen Batch Size, Dropout Rate, dan Learning Rate.
    """
    print("\nLOG: [Tahap 2: Proses Grid Search Otomatis Arsitektur CNN-LSTM]")
    
    X_3D = np.load(os.path.join(DATA_DIR, "X_Raw_ECG_V2.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_Raw_ECG_V2.npy"))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_3D, y, test_size=0.2, random_state=42, stratify=y
    ) 
    
    # Ruang pencarian parameter (Hyperparameter Search Space)
    daftar_batch_size = [16, 32] 
    daftar_dropout = [0.2, 0.4] 
    daftar_learning_rate = [0.001, 0.0005] 
    
    terbaik_akurasi = 0.0 
    terbaik_params = {} 
    
    eksperimen_ke = 1 
    total_eksperimen = len(daftar_batch_size) * len(daftar_dropout) * len(daftar_learning_rate) 
    window_size = X_train.shape[1] 
    
    # Putaran iterasi kombinasi Grid Search
    for bs in daftar_batch_size: 
        for drop in daftar_dropout: 
            for lr in daftar_learning_rate:
                print("\n" + "=" * 55)
                print(f"Eksperimen [{eksperimen_ke}/{total_eksperimen}]") 
                print(f"Menguji Kombinasi -> Batch Size: {bs} | Dropout: {drop} | LR: {lr}") 
                print("=" * 55)
                
                # Mengonstruksi arsitektur CNN-LSTM murni spesifik untuk keperluan Grid Search lokal
                model = Sequential() 
                model.add(Conv1D(filters=32, kernel_size=5, activation="relu", input_shape=(window_size, 1)))
                model.add(MaxPooling1D(pool_size=2)) 
                model.add(Conv1D(filters=64, kernel_size=3, activation="relu")) 
                model.add(MaxPooling1D(pool_size=2)) 
                model.add(LSTM(64)) 
                model.add(Dropout(drop)) 
                model.add(Dense(16, activation="relu")) 
                model.add(Dense(1, activation="sigmoid")) 
                
                optimizer = Adam(learning_rate=lr) 
                model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]) 
                
                # Pelatihan dibatasi hingga 30 Epoch per kombinasi untuk efisiensi komputasi riset
                model.fit(X_train, y_train, epochs=30, batch_size=bs, verbose=1)
                
                # Evaluasi performa test set lokal
                y_pred_prob = model.predict(X_test, verbose=0) 
                y_pred = (y_pred_prob > 0.5).astype(int).flatten() 
                
                akurasi_uji = accuracy_score(y_test, y_pred) 
                print(f" -> Akurasi Uji Kombinasi Ini: {akurasi_uji * 100:.2f}%") 
                
                # Validasi pencapaian rekor parameter terbaik (Best Parameters Tracker)
                if akurasi_uji > terbaik_akurasi: 
                    terbaik_akurasi = akurasi_uji 
                    terbaik_params = {"batch_size": bs, "dropout": drop, "lr": lr} 
                    
                    # Penyimpanan fisik model terbaik hasil pencarian Grid Search ke direktori 03
                    path_save_model = os.path.join(MODEL_SAVE_DIR, "Model_GridSearch_CNNLSTM_Optimal.keras") 
                    model.save(path_save_model) 
                    print(f" -> [REKOR BARU]: File model optimal berhasil diamankan ke direktori.") 
                    
                eksperimen_ke += 1 
                
    # Menampilkan rekapitulasi akhir konfigurasi hyperparameter optimal
    print("\n" + "=" * 55)
    print("=== RINGKASAN REKAPITULASI HASIL AKHIR GRID SEARCH ===")
    print("=" * 55)
    print(f" Konfigurasi Parameter Paling Optimal:")
    print(f"  -> Batch Size    : {terbaik_params.get('batch_size')}") 
    print(f"  -> Dropout Rate  : {terbaik_params.get('dropout')}") 
    print(f"  -> Learning Rate : {terbaik_params.get('lr')}") 
    print(f"  -> Akurasi Uji Terbaik : {terbaik_accuracy * 100:.2f}%")
    print("=" * 55)
    print("LOG: Proses pencarian selesai. Berkas model terbaik siap digunakan.")

def main():
    # 1. Eksekusi Analisis Distribusi dan Baseline Linier (Folder 13)
    analisis_dan_pelatihan_baseline_logreg()
    
    # 2. Eksekusi Optimasi Hyperparameter Komputasional (Folder 14)
    eksekusi_grid_search_cnn_lstm()

if __name__ == "__main__":
    main()