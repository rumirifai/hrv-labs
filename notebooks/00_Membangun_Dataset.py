"""
Fase 00: Pembangunan Dataset Fitur Tabular Statistik (HRV & PRV)
Deskripsi: Membaca raw dataset CLAS, mengekstrak fitur ranah waktu dan frekuensi,
           serta menyimpan hasil ekstraksi ke dalam direktori terproses.
"""

import os
import pandas as pd
import numpy as np
import neurokit2 as nk
from core import loaders

# 1. Konfigurasi Jalur Direktori (Directory Paths)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "00_raw", "CLAS")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "01_processed")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Parameter Sinyal Fisiologis
SAMPLING_RATE = 256
AKTIVITAS_VALID = ['Baseline', 'Neutral', 'Math Test', 'IQ Test', 'Stroop Test']

def tentukan_label_kognitif(aktivitas):
    """Menentukan kelas beban kognitif berdasarkan jenis aktivitas tugas."""
    if aktivitas in ['Math Test', 'IQ Test', 'Stroop Test']:
        return 'Tinggi'
    return 'Rendah'

def main():
    print("LOG: [Memulai Ekstraksi Fitur Tabular Statistik]")
    
    # Memuat seluruh file metadata partisipan
    file_metadata_list = loaders.load_tabular_metadata(RAW_DATA_DIR)
    if not file_metadata_list:
        print("ERROR: Jalur metadata tidak ditemukan. Periksa direktori data/0_raw/CLAS.")
        return

    print(f"LOG: Ditemukan {len(file_metadata_list)} file metadata partisipan. Memulai pemrosesan...")
    
    dataset_final_ecg = []
    dataset_final_ppg = []

    for meta_file in file_metadata_list:
        partisipan_id = os.path.basename(meta_file).split('_')[0]
        
        try:
            df_meta = pd.read_csv(meta_file)
            df_meta.columns = df_meta.columns.str.strip()
            
            # Memfilter aktivitas berdasarkan kriteria riset
            df_meta = df_meta[df_meta['Block Type'].isin(AKTIVITAS_VALID)].copy()
            
            for _, row in df_meta.iterrows():
                jenis_tugas = row['Block Type']
                label = tentukan_label_kognitif(jenis_tugas)
                
                # --- PROSES MODALITAS 1: ECG (Sensor Dada) ---
                kunci_ecg = [x for x in row.keys() if "ECG File" in x]
                if kunci_ecg:
                    nama_file_ecg = str(row[kunci_ecg[0]]).strip()
                    base_prefix_ecg = nama_file_ecg.replace('.csv', '')
                    path_ecg_aktual = loaders.find_actual_signal_file(RAW_DATA_DIR, partisipan_id, base_prefix_ecg)
                    
                    if path_ecg_aktual:
                        sinyal_ecg = loaders.read_raw_csv_signal(path_ecg_aktual, 'ecg2', fallback_index=5)
                        try:
                            sinyal_bersih = nk.ecg_clean(sinyal_ecg, sampling_rate=SAMPLING_RATE)
                            _, info_puncak = nk.ecg_peaks(sinyal_bersih, sampling_rate=SAMPLING_RATE)
                            
                            hrv_time = nk.hrv_time(info_puncak, sampling_rate=SAMPLING_RATE)
                            hrv_freq = nk.hrv_frequency(info_puncak, sampling_rate=SAMPLING_RATE)
                            
                            dataset_final_ecg.append({
                                'Subjek': partisipan_id, 'Tugas': jenis_tugas,
                                'MeanNN_ms': hrv_time.get("HRV_MeanNN").values.item(0),
                                'SDNN_ms': hrv_time.get("HRV_SDNN").values.item(0),
                                'RMSSD_ms': hrv_time.get("HRV_RMSSD").values.item(0),
                                'LF': hrv_freq.get("HRV_LF").values.item(0),
                                'HF': hrv_freq.get("HRV_HF").values.item(0),
                                'LF_HF_Ratio': hrv_freq.get("HRV_LFHF").values.item(0),
                                'Label': label
                            })
                        except Exception:
                            pass

                # --- PROSES MODALITAS 2: PPG (Sensor Tangan) ---
                if 'EDA&PPG File' in row.keys():
                    nama_file_ppg = str(row['EDA&PPG File']).strip()
                    base_prefix_ppg = nama_file_ppg.replace('.csv', '')
                    path_ppg_aktual = loaders.find_actual_signal_file(RAW_DATA_DIR, partisipan_id, base_prefix_ppg)
                    
                    if path_ppg_aktual:
                        sinyal_ppg = loaders.read_raw_csv_signal(path_ppg_aktual, 'ppg', fallback_index=5)
                        try:
                            sinyal_bersih = nk.ppg_clean(sinyal_ppg, sampling_rate=SAMPLING_RATE)
                            info_puncak = nk.ppg_findpeaks(sinyal_bersih, sampling_rate=SAMPLING_RATE)
                            puncak = info_puncak.get("PPG_Peaks")
                            
                            hrv_time = nk.hrv_time(puncak, sampling_rate=SAMPLING_RATE)
                            hrv_freq = nk.hrv_frequency(puncak, sampling_rate=SAMPLING_RATE)
                            
                            dataset_final_ppg.append({
                                'Subjek': partisipan_id, 'Tugas': jenis_tugas,
                                'MeanNN_ms': hrv_time.get("HRV_MeanNN").values.item(0),
                                'SDNN_ms': hrv_time.get("HRV_SDNN").values.item(0),
                                'RMSSD_ms': hrv_time.get("HRV_RMSSD").values.item(0),
                                'LF': hrv_freq.get("HRV_LF").values.item(0),
                                'HF': hrv_freq.get("HRV_HF").values.item(0),
                                'LF_HF_Ratio': hrv_freq.get("HRV_LFHF").values.item(0),
                                'Label': label
                            })
                        except Exception:
                            pass
                            
        except Exception as e:
            print(f"ERROR: Gagal memproses berkas metadata {meta_file}. Pesan: {str(e)}")

    # 3. Penyimpanan Dataset Hasil Ekstraksi
    if dataset_final_ecg:
        pd.DataFrame(dataset_final_ecg).to_csv(os.path.join(OUTPUT_DIR, "Dataset_Fitur_ECG_6_Fitur.csv"), index=False)
        print("LOG: Berhasil menyimpan Dataset_Fitur_ECG_6_Fitur.csv")
    if dataset_final_ppg:
        pd.DataFrame(dataset_final_ppg).to_csv(os.path.join(OUTPUT_DIR, "Dataset_Fitur_PPG_6_Fitur.csv"), index=False)
        print("LOG: Berhasil menyimpan Dataset_Fitur_PPG_6_Fitur.csv")

if __name__ == "__main__":
    main()