"""
Module: core.loaders
Deskripsi: Menyediakan fungsi utilitas untuk pemuatan data (data loading),
           pencarian file pada sistem direktori, dan fusi data multimodal.
"""

import os
import glob
import pandas as pd
import numpy as np

def load_tabular_metadata(base_dataset_dir):
    """
    Melakukan scanning dan memuat seluruh jalur file (file path) metadata partisipan
    yang memiliki sufiks '*Block_Details.csv' di dalam direktori dataset.

    Parameters:
    -----------
    base_dataset_dir : str
        Jalur direktori utama dataset CLAS.

    Returns:
    --------
    list
        Daftar jalur file metadata yang ditemukan.
    """
    search_pattern = os.path.join(base_dataset_dir, "**", "*Block_Details.csv")
    file_metadata_list = glob.glob(search_pattern, recursive=True)
    return file_metadata_list

def find_actual_signal_file(base_dataset_dir, partisipan_id, base_prefix):
    """
    Mencari dan memvalidasi lokasi file sinyal aktual (ECG/PPG) di dalam direktori
    partisipan spesifik untuk menghindari ambiguitas pembacaan lintas subjek.

    Parameters:
    -----------
    base_dataset_dir : str
        Jalur direktori utama dataset CLAS.
    partisipan_id : str
        Identitas unik partisipan (misal: 'Part1').
    base_prefix : str
        Nama depan file sinyal yang dicari tanpa ekstensi.

    Returns:
    --------
    str atau None
        Jalur file aktual jika ditemukan, atau None jika tidak ditemukan.
    """
    search_pattern = os.path.join(base_dataset_dir, "**", partisipan_id, "**", f"{base_prefix}*.csv")
    files_found = glob.glob(search_pattern, recursive=True)
    if len(files_found) > 0:
        return files_found[0]
    return None

def read_raw_csv_signal(path_file, column_name, fallback_index=5):
    """
    Membaca file data mentah fisiologis berbasis CSV dan mengekstrak array sinyal
    berdasarkan nama kolom target atau indeks fallback.

    Parameters:
    -----------
    path_file : str
        Jalur file CSV yang akan dibaca.
    column_name : str
        Nama kolom target (misal: 'ecg2' atau 'ppg').
    fallback_index : int, default=5
        Indeks kolom alternatif jika nama kolom tidak ditemukan pada header.

    Returns:
    --------
    numpy.ndarray
        Array data sinyal mentah.
    """
    df_raw = pd.read_csv(path_file)
    df_raw.columns = df_raw.columns.str.strip()
    
    if column_name in df_raw.columns:
        return df_raw[column_name].values
    else:
        return df_raw.iloc[:, fallback_index].values

def merge_multimodal_features(path_ecg_csv, path_ppg_csv):
    """
    Menggabungkan dataset fitur statistik dari dua modalitas sensor (ECG dan PPG)
    menggunakan metode Inner Join berdasarkan entitas Subjek dan Jenis Tugas.

    Parameters:
    -----------
    path_ecg_csv : str
        Jalur file CSV fitur HRV (ECG).
    path_ppg_csv : str
        Jalur file CSV fitur PRV (PPG).

    Returns:
    --------
    pandas.DataFrame
        DataFrame hasil fusi multimodal dengan penamaan kolom terstandardisasi.
    """
    df_ecg = pd.read_csv(path_ecg_csv)
    df_ppg = pd.read_csv(path_ppg_csv)

    # Sinkronisasi penamaan kolom untuk mencegah bentrokan nama fitur (feature collision)
    df_ecg = df_ecg.rename(columns={
        "MeanNN_ms": "MeanNN_ECG", 
        "SDNN_ms": "SDNN_ECG", 
        "RMSSD_ms": "RMSSD_ECG", 
        "Label": "Label_ECG"
    })
    df_ppg = df_ppg.rename(columns={
        "MeanNN_ms": "MeanNN_PPG", 
        "SDNN_ms": "SDNN_PPG", 
        "RMSSD_ms": "RMSSD_PPG", 
        "Label": "Label_PPG"
    })

    df_multi = pd.merge(df_ecg, df_ppg, on=["Subjek", "Tugas"], how="inner")
    return df_multi