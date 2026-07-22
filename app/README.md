# Dashboard Visualisasi Penelitian HRV
## Sistem Deteksi Beban Kognitif Mahasiswa

**Muhamad Rumi Rifai | 140810220026 | Teknik Informatika UNPAD 2025**

---

## Cara Menjalankan (Localhost)

### 1. Install Dependencies

Pastikan Python 3.9+ sudah terinstall, lalu jalankan di terminal dari folder `app/`:

```bash
pip install -r requirements_app.txt
```

### 2. Jalankan Server Flask

```bash
python app.py
```

Server akan berjalan di: **http://localhost:5000**

Buka browser dan akses URL tersebut.

---

## Fitur Aplikasi

| Tab | Isi |
|:---|:---|
| 🗄️ **Dataset** | Informasi dataset CLAS, distribusi kelas, statistik fitur HRV, pratinjau 20 baris data |
| 📊 **Preprocessing** | Pipeline visualisasi, upload file CSV sinyal mentah ECG/PPG, grafik sinyal interaktif |
| 🏆 **Semua Model** | Master leaderboard 16 model, confusion matrix, grafik komparasi akurasi |
| 🔮 **Demo Prediksi** | Input manual HRV dengan slider + tooltip, preset demo dari data CLAS, prediksi real-time |

---

## Struktur Folder

```
app/
├── app.py                  ← Flask backend (API endpoints)
├── requirements_app.txt    ← Dependencies
├── README.md               ← Panduan ini
├── data/
│   ├── dataset_ecg.csv     ← Dataset fitur HRV ECG
│   ├── dataset_ppg.csv     ← Dataset fitur HRV PPG
│   └── rekap_model.csv     ← Data performa semua model
├── models/
│   └── model_rank1_ppg.keras  ← Model CNN-LSTM PPG Tabular (Rank #1)
├── static/
│   ├── css/style.css       ← Stylesheet dark mode
│   ├── js/dashboard.js     ← Logika frontend + Chart.js
│   └── img/                ← Gambar dari laporan penelitian
└── templates/
    └── index.html          ← Halaman utama dashboard
```

---

## API Endpoints

| Method | Endpoint | Deskripsi |
|:---|:---|:---|
| GET | `/api/dataset/ecg` | Data statistik dataset ECG |
| GET | `/api/dataset/ppg` | Data statistik dataset PPG |
| GET | `/api/leaderboard` | Performa semua 16 model |
| POST | `/api/predict` | Prediksi dari nilai HRV manual |
| POST | `/api/process-signal` | Proses file CSV sinyal mentah |

---

## Model yang Digunakan

**CNN-LSTM Baseline PPG Tabular** (Rank #1 Master Leaderboard)
- Akurasi: **65,17%** | F1-Score: **64,16%**
- Input: 6 fitur HRV dari sinyal PPG (MeanNN, SDNN, RMSSD, LF, HF, LF/HF)
- File: `models/model_rank1_ppg.keras`

---

## Dataset

CLAS (Cognitive Load, Affect, and Stress) Dataset
- 60 partisipan, 5 jenis tugas kognitif
- Label: Beban Rendah (Baseline + Neutral) / Beban Tinggi (Math, Stroop, IQ Test)
- Sensor: ECG (chest) + PPG (fingertip)
