"""
Fase 02: Pembangunan Dataset Waveform (Gelombang Mentah) via Sliding Window
Deskripsi: Mengekstrak potongan sinyal kontinu ECG menjadi data tensor runtun-waktu (time-series)
           menggunakan metode jendela bergeser untuk kebutuhan pemodelan arsitektur Deep Learning.
"""

import os
import numpy as np
import pandas as pd
from core import loaders, processors

# 1. Konfigurasi Jalur Direktori (Directory Paths)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "00_raw", "CLAS", "Participants")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "02_waveform")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Parameter Frekuensi Sampling (Sampling Rate)
FS = 256

# 3. Klasifikasi Tugas Berdasarkan Protokol Dataset CLAS
TUGAS_RENDAH = tuple(("1_ecg", "4_ecg", "7_ecg", "10_ecg", "12_ecg", "14_ecg", "16_ecg", 
                      "18_ecg", "20_ecg", "23_ecg", "28_ecg", "30_ecg", "32_ecg", "34_ecg", "36_ecg"))
TUGAS_TINGGI = tuple(("2_ecg", "5_ecg", "8_ecg"))

def ekstraksi_dataset_waveform(window_sec, step_sec, versi_label):
    """
    Melakukan scanning direktori partisipan dan mengekstrak segmen gelombang ECG
    berdasarkan durasi jendela waktu dan pergeseran yang ditentukan.
    """
    print(f"\nLOG: [Memulai Ekstraksi Sinyal Waveform ECG - Versi {versi_label}]")
    print(f"     Parameter: Window = {window_sec} detik | Step = {step_sec} detik")
    
    if not os.path.exists(RAW_DATA_DIR):
        print(f"ERROR: Direktori raw data tidak ditemukan pada {RAW_DATA_DIR}")
        return

    subjek_list = os.listdir(RAW_DATA_DIR)
    X_master = []
    y_master = []
    total_sampel_terekstrak = 0

    for subjek in subjek_list:
        path_by_block = os.path.join(RAW_DATA_DIR, subjek, "by_block")
        if not os.path.isdir(path_by_block):
            continue
            
        file_ecg_list = os.listdir(path_by_block)
        for nama_file in file_ecg_list:
            # Determinasi label target berdasarkan prefix nama file
            if nama_file.startswith(TUGAS_RENDAH):
                label = 0
            elif nama_file.startswith(TUGAS_TINGGI):
                label = 1
            else:
                continue
                
            path_file = os.path.join(path_by_block, nama_file)
            
            try:
                # Membaca array sinyal dari kolom 'ecg2' melalui fungsi modular loaders
                sinyal_ecg = loaders.read_raw_csv_signal(path_file, column_name='ecg2', fallback_index=5)
                
                # Memproses segmentasi jendela bergeser melalui fungsi modular processors
                X_potongan, y_potongan = processors.slice_signal_sliding_window(
                    sinyal=sinyal_ecg,
                    window_sec=window_sec,
                    step_sec=step_sec,
                    fs=FS,
                    label=label
                )
                
                X_master.extend(X_potongan)
                y_master.extend(y_potongan)
                total_sampel_terekstrak += len(X_potongan)
                
            except Exception as e:
                print(f"PERINGATAN: Gagal memproses file {nama_file} pada subjek {subjek}. Pesan: {str(e)}")

    print(f"LOG: Ekstraksi Selesai. Total segmen gelombang terekstrak: {total_sampel_terekstrak}")

    # 4. Transformasi ke Format Tensor NumPy dan Penyimpanan Fisik Berkas
    if total_sampel_terekstrak > 0:
        X_array = np.array(X_master)
        y_array = np.array(y_master)

        # Reshape dimensi ke format Keras Tensor: (jumlah_sampel, timestep, jumlah_fitur/channel)
        window_size = int(FS * window_sec)
        X_tensor = np.reshape(X_array, (len(X_array), window_size, 1))

        print(f" -> Matriks X Dimensi: {X_tensor.shape}")
        print(f" -> Matriks Y Dimensi: {y_array.shape}")

        np.save(os.path.join(OUTPUT_DIR, f"X_Raw_ECG_{versi_label}.npy"), X_tensor)
        np.save(os.path.join(OUTPUT_DIR, f"y_Raw_ECG_{versi_label}.npy"), y_array)
        print(f"LOG: Berhasil menyimpan berkas dataset Versi {versi_label} ke {OUTPUT_DIR}")
    else:
        print(f"ERROR: Proses gagal, tidak ada data gelombang yang terekstrak untuk Versi {versi_label}.")

def main():
    # 1. Ekstraksi Strategi 1 / V2 (Jendela 5 Detik, Pergeseran 2.5 Detik)
    ekstraksi_dataset_waveform(window_sec=5, step_sec=2.5, versi_label="V2")
    
    # 2. Ekstraksi Strategi 1.5 / V3 (Jendela 10 Detik, Pergeseran 5 Detik)
    ekstraksi_dataset_waveform(window_sec=10, step_sec=5, versi_label="V3")

if __name__ == "__main__":
    main()