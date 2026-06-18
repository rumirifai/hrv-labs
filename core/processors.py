"""
Module: core.processors
Deskripsi: Menyediakan fungsi analitik untuk pemrosesan sinyal digital (DSP),
           rekayasa fitur (feature engineering), seleksi fitur statistik,
           serta kalkulasi interpretabilitas model White-Box.
"""

import numpy as np
import pandas as pd
from scipy.signal import detrend
from scipy.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LinearRegression

def inspect_data_balance(y_labels):
    """
    Melakukan analisis statistik deskriptif terhadap distribusi frekuensi
    dan proporsi persentase dari label/kelas target di dalam dataset.

    Parameters:
    -----------
    y_labels : array-like
        Array berisi label target dari dataset.

    Returns:
    --------
    dict
        Kamus frekuensi kemunculan tiap kelas target.
    """
    total_data = len(y_labels)
    unique, counts = np.unique(y_labels, return_counts=True)
    distribution = dict(zip(unique, counts))
    
    print("LOG: [Analisis Distribusi Data Target]")
    print(f"Total Sampel Dataset : {total_data} baris")
    for label, count in distribution.items():
        percentage = (count / total_data) * 100
        print(f" -> Kelas [{label}] : {count} sampel ({percentage:.2f}%)")
    
    if len(distribution) == 2 and distribution.get(0) != distribution.get(1):
        print("PERINGATAN: Terdeteksi ketidakseimbangan kelas (Class Imbalance).")
        print("            Prioritaskan evaluasi metrik Recall dan F1-Score.")
    print("-" * 40)
    return distribution

def apply_moving_average_filter(signal_array, window_size=5):
    """
    Mereduksi komponen noise berfrekuensi tinggi menggunakan Moving Average Filter
    melalui operasi konvolusi satu dimensi diskrit.

    Parameters:
    -----------
    signal_array : numpy.ndarray
        Array sinyal digital input.
    window_size : int, default=5
        Lebar jendela tapis (filter window width).

    Returns:
    --------
    numpy.ndarray
        Array sinyal hasil penyaringan (smoothed signal).
    """
    filter_weights = np.divide(np.ones(window_size), window_size)
    return np.convolve(signal_array, filter_weights, mode="same")

def slice_signal_sliding_window(sinyal, window_sec, step_sec, fs, label):
    """
    Segmentasi sinyal kontinu menggunakan metode Jendela Bergeser (Sliding Window).
    Setiap segmen diolah melalui tahapan Detrending, Smoothing Filter, 
    dan Standardisasi Z-Score secara terisolasi.

    Parameters:
    -----------
    sinyal : numpy.ndarray
        Array sinyal waveform mentah.
    window_sec : float
        Durasi panjang jendela segmentasi dalam satuan detik.
    step_sec : float
        Durasi pergeseran jendela (stride) dalam satuan detik.
    fs : int
        Frekuensi sampling sinyal (Sampling Rate dalam Hz).
    label : int
        Label target kognitif untuk segmen sinyal terkait.

    Returns:
    --------
    tuple (list, list)
        Sepasang daftar (list) segmen data matriks X dan label target y.
    """
    window_size = int(fs * window_sec)
    step_size = int(fs * step_sec)
    panjang_sinyal = len(sinyal)
    
    X_potongan = []
    y_potongan = []
    
    for i in range(0, panjang_sinyal - window_size + 1, step_size):
        potongan_mentah = np.take(sinyal, range(i, i + window_size))
        
        # 1. Detrending untuk mereduksi efek baseline wander akibat respirasi
        potongan_detrend = detrend(potongan_mentah)
        
        # 2. Moving Average Filter untuk mereduksi noise frekuensi tinggi
        potongan_smooth = apply_moving_average_filter(potongan_detrend, window_size=5)
        
        # 3. Standardisasi Z-Score segmen lokal
        rata_rata = np.mean(potongan_smooth)
        std_dev = np.std(potongan_smooth)
        
        if std_dev == 0:
            continue
            
        potongan_final = np.divide(np.subtract(potongan_smooth, rata_rata), std_dev)
        
        X_potongan.append(potongan_final)
        y_potongan.append(label)
        
    return X_potongan, y_potongan

def compute_ufr_weights(X, y):
    """
    Menghitung skor Univariate Feature Ranking (UFR) berbasis uji ANOVA F-Value
    dan mentransformasikannya menjadi bobot persentase normalisasi.

    Parameters:
    -----------
    X : numpy.ndarray
        Matriks fitur input.
    y : numpy.ndarray
        Array label target.

    Returns:
    --------
    numpy.ndarray
        Array bobot normalisasi untuk tiap fitur.
    """
    selector = SelectKBest(score_func=f_classif, k="all")
    selector.fit(X, y)
    scores = selector.scores_
    
    total_skor = np.sum(scores)
    ufr_weights = np.divide(scores, total_skor)
    return ufr_weights

def compute_mrmr_weights(X, y, feature_names):
    """
    Menghitung bobot kepentingan fitur berdasarkan prinsip kriteria 
    Minimum Redundancy Maximum Relevance (MRMR).

    Parameters:
    -----------
    X : numpy.ndarray
        Matriks fitur input.
    y : numpy.ndarray
        Array label target.
    feature_names : list
        Daftar nama fitur sesuai urutan kolom pada matriks X.

    Returns:
    --------
    numpy.ndarray
        Array distribusi bobot normalisasi berbasis MRMR.
    """
    # 1. Kalkulasi komponen Maksimum Relevansi via ANOVA F-Value
    selector = SelectKBest(score_func=f_classif, k="all")
    selector.fit(X, y)
    relevance = selector.scores_
    
    # 2. Kalkulasi komponen Minimum Redundansi via Matriks Korelasi Pearson Absolut
    df_fitur = pd.DataFrame(X, columns=feature_names)
    corr_matrix = df_fitur.corr().abs().values
    
    mrmr_scores = []
    num_features = len(feature_names)
    
    for i in range(num_features):
        list_korelasi_lain = [corr_matrix.item((i, j)) for j in range(num_features) if i != j]
        redundancy_avg = np.mean(list_korelasi_lain) if len(list_korelasi_lain) > 0 else 0.0
        
        # Penambahan nilai epsilon (0.0001) untuk mencegah pembagian dengan nol (zero-division protection)
        score_mrmr = relevance.item(i) / (redundancy_avg + 0.0001)
        mrmr_scores.append(score_mrmr)
        
    total_mrmr = np.sum(mrmr_scores)
    mrmr_weights = np.divide(np.array(mrmr_scores), total_mrmr)
    return mrmr_weights

def print_kfold_evaluation_summary(acc_list, prec_list, rec_list, f1_list, phase_name=""):
    """
    Mencetak rekapitulasi rata-rata metrik evaluasi performa klasifikasi 
    dari pengujian Stratified K-Fold Cross Validation.

    Parameters:
    -----------
    acc_list, prec_list, rec_list, f1_list : list
        Kumpulan nilai metrik evaluasi per lipatan (fold).
    phase_name : str, optional
        Nama fase eksperimen sebagai identifikasi log.
    """
    print(f"\nLOG: [Hasil Evaluasi Rata-rata K-Fold - {phase_name.upper()}]")
    print(f" -> Rata-rata Akurasi   : {np.mean(acc_list) * 100:.2f}%")
    print(f" -> Rata-rata Precision : {np.mean(prec_list) * 100:.2f}%")
    print(f" -> Rata-rata Recall    : {np.mean(rec_list) * 100:.2f}%")
    print(f" -> Rata-rata F1-Score  : {np.mean(f1_list) * 100:.2f}%")
    print("-" * 50)

def extract_whitebox_framework(model, X_scaled, feature_names, is_temporal=True):
    """
    Framework Global: Mentransformasikan model inferensi Black-Box (Keras Deep AI)
    menjadi persamaan linier matematika transparan (White-Box Equation) melalui 
    metode pendekatan optimasi numerik linier (Singh & Bura, 2025).

    Parameters:
    -----------
    model : tensorflow.keras.Model atau tensorflow.keras.Sequential
        Model Black-Box terlatih yang akan diekstrak parameternya.
    X_scaled : numpy.ndarray
        Matriks data fitur input yang telah terstandardisasi.
    feature_names : list
        Daftar string nama fitur yang diuji.
    is_temporal : bool, default=True
        Flag penentu format dimensi input jaringan runtun-waktu (3D Tensor).

    Returns:
    --------
    tuple (dict, float)
        Kamus koefisien bobot parameter (weights) fitur dan nilai bias konstanta.
    """
    # 1. Penyesuaian dimensi input untuk inferensi model Deep Learning (Format 3D Keras)
    if is_temporal and len(X_scaled.shape) == 2:
        X_input = np.reshape(X_scaled, (len(X_scaled), X_scaled.shape[1], 1))
    else:
        X_input = X_scaled
        
    probabilitas_ai = model.predict(X_input, verbose=0).flatten()
    
    # 2. Penyesuaian dimensi data fit untuk estimasi Linear Regression (Format 2D)
    if len(X_scaled.shape) == 3:
        X_scaled_2d = np.squeeze(X_scaled)
    else:
        X_scaled_2d = X_scaled

    # 3. Prosedur Optimasi Numerik Linier Kontinu
    regressor = LinearRegression()
    regressor.fit(X_scaled_2d, probabilitas_ai)
    
    # 4. Konstruksi Persamaan Matematika dan Analisis Magnitudo Kontribusi Fitur
    bobot_fitur = {}
    print("\n" + "="*60)
    print("🎓 FORMULASI INDEKS BEBAN KOGNITIF (UNIVERSAL WHITE-BOX) 🎓")
    print("="*60)
    print("Indeks SI = ")
    
    for i, name in enumerate(feature_names):
        weight = regressor.coef_.item(i)
        bobot_fitur[name] = weight
        print(f"   ({weight:+.4f} * {name}_Standar) ")
        
    bias = regressor.intercept_
    print(f"   ({bias:+.4f} [Bias Konstanta])")
    print("="*60)
    
    print("\n📊 PERINGKAT MAGNITUDO PENGARUH FITUR (NILAI MUTLAK COEF) 📊")
    sorted_features = sorted(bobot_fitur.items(), key=lambda item: abs(item[1]), reverse=True)
    for rank, (name, val) in enumerate(sorted_features, 1):
        arah = "Positif (Berbanding Lurus)" if val >= 0 else "Negatif (Berbanding Terbalik)"
        print(f"  {rank}. {name:<15} : Nilai Mutlak = {abs(val):.4f} | Arah Kontribusi: {arah}")
    print("="*60)
    
    return bobot_fitur, bias