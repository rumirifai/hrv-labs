"""
Fase 03: Pemodelan Berbasis Sinyal Waveform (Deep Learning Waveform Processing)
Deskripsi: Melatih arsitektur tingkat tinggi CNN-BiLSTM menggunakan dataset runtun-waktu mentah.
           Mengimplementasikan skema pelatihan penuh (Full Epoch), penjadwalan laju belajar Cosine Decay,
           serta validasi performa menggunakan 5-Fold Stratified Cross Validation.
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import ModelCheckpoint

from core import processors, models

# 1. Konfigurasi Jalur Direktori
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "02_waveform")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "data", "03_saved_models")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def jalankan_pelatihan_bilstm_v2():
    """
    Menjalankan pelatihan penuh 100 Epoch pada dataset V2 (Jendela 5 Detik)
    menggunakan trik restorasi bobot optimal pasca-pelatihan tanpa Early Stopping.
    """
    print("\nLOG: [Memulai Pelatihan CNN-BiLSTM - Strategi 1 (Dataset V2)]")
    
    X = np.load(os.path.join(DATA_DIR, "X_Raw_ECG_V2.npy")) 
    y = np.load(os.path.join(DATA_DIR, "y_Raw_ECG_V2.npy")) 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 
    processors.inspect_data_balance(y_train)
    
    window_size = X_train.shape[1]
    model = models.build_cnn_bilstm_waveform(window_size=window_size, learning_rate=0.001) 
    
    # Perhitungan pembobotan kelas penyeimbang data
    cls_w = class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train) 
    dict_cls_w = {0: cls_w.item(0), 1: cls_w.item(1)} 
    
    # Callback Penyimpan Model Terbaik secara internal (ModelCheckpoint)
    checkpoint_path = os.path.join(MODEL_SAVE_DIR, "Model_Terbaik_BiLSTM_V2_FullEpoch.keras") 
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1
    ) 
    
    # Pelatihan penuh dijalankan 100 epoch tanpa interupsi Early Stopping
    model.fit(
        X_train, y_train, epochs=100, batch_size=16, class_weight=dict_cls_w,
        validation_data=(X_test, y_test), callbacks=[checkpoint_callback], verbose=1
    ) 
    
    # Restorasi bobot paling optimal yang tersimpan secara manual
    print("LOG: Melakukan restorasi bobot parameter terbaik ke dalam arsitektur model...")
    model.load_weights(checkpoint_path) 
    
    y_pred = (model.predict(X_test, verbose=1) > 0.5).astype(int).flatten() 
    
    print("\n--- HASIL EVALUASI STRATEGI 1 (DATA V2) ---")
    print(f"Akurasi   : {accuracy_score(y_test, y_pred) * 100:.2f}%") 
    print(f"Precision : {precision_score(y_test, y_pred, zero_division=0) * 100:.2f}%") 
    print(f"Recall    : {recall_score(y_test, y_pred, zero_division=0) * 100:.2f}%") 
    print(f"F1-Score  : {f1_score(y_test, y_pred, zero_division=0) * 100:.2f}%") 
    print("=" * 45)

def jalankan_pelatihan_bilstm_v3():
    """
    Menjalankan pelatihan model pada dataset V3 (Jendela 10 Detik) teroptimasi 
    menggunakan mekanisme Cosine Learning Rate Scheduler.
    """
    print("\nLOG: [Memulai Pelatihan CNN-BiLSTM - Strategi 1.5 (Dataset V3)]")
    
    X = np.load(os.path.join(DATA_DIR, "X_Raw_ECG_V3.npy")) 
    y = np.load(os.path.join(DATA_DIR, "y_Raw_ECG_V3.npy")) 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 
    
    # Kalkulasi total langkah optimasi untuk kebutuhan Cosine Decay
    langkah_per_epoch = int(np.ceil(len(X_train) / 16.0)) 
    total_langkah_decay = langkah_per_epoch * 100 
    
    window_size = X_train.shape[1]
    model = models.build_cnn_bilstm_waveform(
        window_size=window_size, learning_rate=0.001, 
        use_cosine_decay=True, total_steps=total_langkah_decay
    ) 
    
    cls_w = class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train) 
    dict_cls_w = {0: cls_w.item(0), 1: cls_w.item(1)} 
    
    checkpoint_path = os.path.join(MODEL_SAVE_DIR, "Model_Terbaik_BiLSTM_V3_CosineLR.keras") 
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1
    ) 
    
    model.fit(
        X_train, y_train, epochs=100, batch_size=16, class_weight=dict_cls_w,
        validation_data=(X_test, y_test), callbacks=[checkpoint_callback], verbose=1
    ) 
    
    print("LOG: Melakukan restorasi bobot parameter terbaik (Dataset V3)...")
    model.load_weights(checkpoint_path) 
    
    y_pred = (model.predict(X_test, verbose=1) > 0.5).astype(int).flatten() 
    
    print("\n--- HASIL EVALUASI STRATEGI 1.5 (DATA V3) ---")
    print(f"Akurasi   : {accuracy_score(y_test, y_pred) * 100:.2f}%") 
    print(f"Precision : {precision_score(y_test, y_pred, zero_division=0) * 100:.2f}%") 
    print(f"Recall    : {recall_score(y_test, y_pred, zero_division=0) * 100:.2f}%") 
    print(f"F1-Score  : {f1_score(y_test, y_pred, zero_division=0) * 100:.2f}%") 
    print("=" * 45)

def jalankan_validasi_akademis_5fold():
    """
    Melakukan pengujian validasi akhir model CNN-BiLSTM pada dataset V2 
    menggunakan skema 5-Fold Stratified Cross Validation untuk menjamin stabilitas performa.
    """
    print("\nLOG: [Memulai Validasi Akademis Akhir - 5-Fold Cross Validation (Dataset V2)]")
    
    X = np.load(os.path.join(DATA_DIR, "X_Raw_ECG_V2.npy")) 
    y = np.load(os.path.join(DATA_DIR, "y_Raw_ECG_V2.npy")) 
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 
    acc_l, prec_l, rec_l, f1_l = [], [], [], []
    fold_ke = 1
    
    for train_index, test_index in skf.split(X, y):
        print(f"\n--- MENGEKSEKUSI VALIDASI FOLD KE-{fold_ke} ---")
        X_train, X_test = np.take(X, train_index, axis=0), np.take(X, test_index, axis=0)  
        y_train, y_test = np.take(y, train_index, axis=0), np.take(y, test_index, axis=0) 
        
        window_size = X_train.shape[1]
        model = models.build_cnn_bilstm_waveform(window_size=window_size, learning_rate=0.001) 
        
        cls_w = class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)  
        dict_cls_w = {0: cls_w.item(0), 1: cls_w.item(1)} 
        
        fold_model_path = os.path.join(MODEL_SAVE_DIR, f"Model_Terbaik_Fold_{fold_ke}.keras") 
        checkpoint_callback = ModelCheckpoint(
            filepath=fold_model_path, monitor="val_loss", save_best_only=True, verbose=1
        )  
        
        model.fit(
            X_train, y_train, epochs=100, batch_size=16, class_weight=dict_cls_w,
            validation_data=(X_test, y_test), callbacks=[checkpoint_callback], verbose=1
        )  
        
        # Muat ulang otak paling optimal khusus pada fold ini
        model.load_weights(fold_model_path) 
        
        y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten() 
        
        acc_l.append(accuracy_score(y_test, y_pred))
        prec_l.append(precision_score(y_test, y_pred, zero_division=0)) 
        rec_l.append(recall_score(y_test, y_pred, zero_division=0)) 
        f1_l.append(f1_score(y_test, y_pred, zero_division=0)) 
        
        fold_ke += 1
        
    # Menampilkan rekapitulasi rata-rata metrik akhir lintas lipatan (fold) via core.processors
    processors.print_kfold_evaluation_summary(acc_l, prec_l, rec_l, f1_l, phase_name="5-Fold Waveform Final") 

def main():
    # 1. Eksperimen Strategi 1 Jendela 5 Detik Full Epoch (Folder 10)
    jalankan_pelatihan_bilstm_v2()
    
    # 2. Eksperimen Strategi 1.5 Jendela 10 Detik + Cosine LR Scheduler (Folder 11)
    jalankan_pelatihan_bilstm_v3()
    
    # 3. Validasi Akademis Akhir 5-Fold Cross Validation (Folder 12)
    jalankan_validasi_akademis_5fold()

if __name__ == "__main__":
    main()