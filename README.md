# 🫀 HRV-Labs: Klasifikasi Beban Kognitif Berbasis Sensor Detak Jantung

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1.3-black?logo=flask)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange?logo=tensorflow)](https://tensorflow.org/)

**Skripsi S-1 Teknik Informatika — Universitas Padjadjaran 2026**

</div>

---

## 📌 Tentang Proyek Ini

Repositori ini merupakan implementasi lengkap penelitian skripsi berjudul:

> **"Klasifikasi Beban Kognitif pada Pembelajaran Daring Berbasis Sensor Detak Jantung Menggunakan Convolutional Neural Network–Long Short-Term Memory"**

Penelitian ini mengklasifikasikan level **beban kognitif** (Rendah vs Tinggi) dari data sinyal fisiologis detak jantung (ECG & PPG) menggunakan arsitektur hibrida **CNN-LSTM** dan variasinya, dengan kerangka metodologi **Knowledge Discovery in Databases (KDD)**.

---

## 🔬 Latar Belakang

Pembelajaran daring menuntut adaptasi terhadap teknologi, materi visual, dan regulasi diri secara simultan. Kondisi ini berpotensi memicu **cognitive overload** yang menghambat pemahaman materi. Metode pengukuran tradisional seperti kuesioner tidak mampu memantau beban kognitif secara *real-time* dan objektif.

Penelitian ini memanfaatkan sinyal **ECG (Electrocardiography)** dan **PPG (Photoplethysmography)** dari dataset sekunder [CLAS](https://doi.org/10.1109/BIA48344.2019.8967457) sebagai dasar ekstraksi fitur **Heart Rate Variability (HRV)** untuk keperluan klasifikasi otomatis.

---

## 🏆 Hasil Terbaik (Leaderboard 16 Model)

Perangkingan menggunakan standar **F1-Score First → Accuracy** (karena kondisi data imbalanced).

| Rank | Model | Sensor | Accuracy | F1-Score |
|:---:|:---|:---|:---:|:---:|
| 🥇 1 | CNN-LSTM Hyperparam Tuned | ECG Waveform | 65,07% | **71,07%** |
| 🥈 2 | CNN-BiLSTM V2 | ECG Waveform | 52,25% | 68,63% |
| 🥉 3 | CNN-BiLSTM V3 | ECG Waveform | 64,87% | 67,29% |
| 4 | CNN-LSTM + UFR | PPG Tabular | 50,56% | 66,15% |
| 5 | CNN-LSTM + mRMR | PPG Tabular | 50,00% | 66,03% |
| — | *...11 model lainnya* | — | — | — |

📊 Lihat tabel lengkap di [TABEL_RINGKASAN_SKRIPSI.md](./TABEL_RINGKASAN_SKRIPSI.md)

---

## 🗂️ Struktur Repositori

```
hrv-labs/
│
├── 📁 app/                         # Dashboard Web Interaktif (Flask)
│   ├── app.py                      # Backend Flask — API endpoints
│   ├── data/
│   │   └── rekap_model.csv         # Database 16 model (master leaderboard)
│   ├── static/
│   │   ├── css/style.css           # Stylesheet UI
│   │   └── js/dashboard.js         # Logika frontend leaderboard & chart
│   └── templates/index.html        # Template UI dashboard
│
├── 📁 core/                        # Library Modular Utama
│   ├── loaders.py                  # Pembacaan sinyal & fusi dataset multimodal
│   ├── models.py                   # Definisi arsitektur CNN-LSTM, BiLSTM, Attention
│   └── processors.py               # DSP, segmentasi, UFR, mRMR, White-Box Approximator
│
├── 📁 notebooks/                   # Pipeline Eksperimen End-to-End
│   ├── 00_Membangun_Dataset.py     # Ekstraksi fitur HRV tabular dari CLAS
│   ├── 01_Fase_Statistik.py        # Evaluasi SVM, RF, CNN-LSTM (10-Fold CV)
│   ├── 02_Membangun_Dataset_Waveform.py  # Segmentasi sliding window .npy
│   ├── 03_Fase_Waveform_DeepLearning.py  # Pelatihan CNN-LSTM Waveform
│   ├── 04_Hyperparameter_Tuning.py # Grid Search CNN-LSTM
│   ├── 05_Multimodal.py            # Fusi ECG+PPG Multimodal Expert
│   ├── 06_White_Box_Approximator.py # Surrogate model & analisis kontribusi fitur
│   └── 07_Sistem_Rekapitulasi_Eksperimen.py  # Generator tabel master rekap
│
├── 📁 data/                        # Dataset (tidak disertakan — lihat catatan)
│   ├── 00_raw/                     # Data mentah CLAS (.mat)
│   ├── 01_processed/               # CSV tabular & CSV rekap akurasi
│   └── 02_waveform/                # Array waveform .npy
│
├── 📁 reports/                     # Gambar & output laporan skripsi
├── 📁 Dokumen/                     # Naskah skripsi & PPT sidang (PDF)
│
├── TABEL_RINGKASAN_SKRIPSI.md      # Leaderboard master 16 model
├── requirements.txt                # Daftar dependensi Python
└── README.md                       # Dokumen ini
```

---

## 🚀 Menjalankan Dashboard Web Secara Lokal

### Prasyarat
- Python 3.13+
- pip

### Instalasi

`ash
# 1. Clone repositori
git clone https://github.com/rumirifai/hrv-labs.git
cd hrv-labs

# 2. Install dependensi
pip install -r requirements.txt

# 3. Jalankan server Flask
py app/app.py
`

### Akses Dashboard

Buka peramban web dan akses: http://localhost:5000

### Fitur Dashboard

| Tab | Deskripsi |
|:---|:---|
| 📊 **Dataset** | Statistik deskriptif & distribusi kelas ECG/PPG Tabular |
| 🔬 **Preprocessing** | Demo interaktif pemrosesan sinyal ECG/PPG dari dataset CLAS |
| 🏆 **Model Leaderboard** | Perbandingan performa 16 model (F1-Score bar chart) |
| 🎯 **Demo Prediksi** | Inferensi real-time via 6 slider fitur HRV tabular (PPG) |
| 👥 **Credits** | Informasi peneliti & tim pembimbing |

---

## 📦 Dependensi Utama

| Library | Versi | Fungsi |
|:---|:---:|:---|
| lask | 3.1.3 | Backend web server |
| 	ensorflow | 2.18 | Deep learning (CNN-LSTM) |
| scikit-learn | 1.6 | SVM, RF, StandardScaler, StratifiedKFold |
| 
eurokit2 | 0.2.11 | Pemrosesan sinyal ECG/PPG & ekstraksi HRV |
| pandas | 2.2 | Manipulasi data tabular |
| 
umpy | 1.26 | Komputasi numerik & array waveform |

---

## ⚠️ Catatan Dataset

Dataset **CLAS (Cognitive Load, Affect and Stress)** adalah dataset sekunder publik dari:

> Markova, V., Ganchev, T., & Kalinkov, K. (2019). CLAS: A Database for Cognitive Load, Affect and Stress Recognition. *Proceedings of BIA 2019*. https://doi.org/10.1109/BIA48344.2019.8967457

Dataset mentah (.mat) **tidak disertakan** dalam repositori ini karena ukurannya yang besar (>2 GB). Silakan unduh langsung dari sumber resminya.

---

## 👥 Kredit

| Peran | Nama |
|:---|:---|
| **Peneliti Utama** | Muhamad Rumi Rifai (NPM 140810220026) |
| **Pembimbing Utama** | Dr. Mira Suryani, S.Pd., M.Kom. (NIP. 19891230 201504 2 001) |
| **Co. Pembimbing** | Dr. Juli Rejito, M.Kom. (NIP. 19680717 199303 1 003) |
| **Institusi** | Laboratorium HCI, S-1 Teknik Informatika, FMIPA — Universitas Padjadjaran |
| **Tahun** | 2026 |

---

<div align="center">
<sub>S-1 Teknik Informatika · Universitas Padjadjaran · 2026</sub>
</div>
