"""
Sistem Rekapitulasi Eksperimen Skripsi (Versi Standar Akademis Murni V6)
Deskripsi: Membaca database master CSV, mengaudit status keseimbangan data asli,
           mencatat mekanisme balancing murni dari kode sumber eksperimen,
           serta menerbitkan papan peringkat komparasi standar internasional (IEEE/Elsevier)
           berbasis F1-Score TANPA kolom status/keterangan kustom dan tanpa catatan kaki.
"""

import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tabulate import tabulate

from core import processors

DATA_TABULAR = os.path.join(BASE_DIR, "data", "01_processed")

# Pemetaan Eksplisit Mekanisme Penyeimbangan Murni dari Kode Sumber Eksperimen
MEKANISME_BALANCING_MAP = {
    "CNN-LSTM Hyperparam Tuned": "Tanpa Intervensi / Grid Search (Unweighted)",
    "CNN-LSTM Baseline": "Class Weights (Algorithmic Cost)",
    "CNN-BiLSTM V3": "Class Weights + Cosine LR Scheduler",
    "CNN-BiLSTM V2": "Class Weights (Algorithmic Cost)",
    "LSTM + Attention": "Class Weights (Algorithmic Cost)",
    "CNN-LSTM + UFR": "Class Weights + Feature Weighting (UFR)",
    "CNN-LSTM + mRMR": "Class Weights + Feature Weighting (mRMR)",
    "SVM Baseline": "Class Weights (Sklearn Balanced)",
    "Random Forest Baseline": "Class Weights (Sklearn Balanced)",
    "Fusi Multimodal Expert": "Class Weights (Hybrid Context)"
}


def tentukan_mekanisme_balancing(model_name):
    """Mengembalikan mekanisme penyeimbangan presisi dari kode sumber eksperimen."""
    return MEKANISME_BALANCING_MAP.get(str(model_name), "Class Weights (Algorithmic Cost)")


def main():
    print("LOG: [Membangun Tabel Ringkasan Eksperimen Skripsi - Standar Murni V6]")

    csv_path = os.path.join(DATA_TABULAR, "REKAP_AKURASI_MASTER.csv")
    md_output = os.path.join(BASE_DIR, "TABEL_RINGKASAN_SKRIPSI.md")

    if not os.path.exists(csv_path):
        print(f"ERROR: Berkas database master tidak ditemukan di: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    status_list, rasio_list, sumber_list = [], [], []
    for _, row in df.iterrows():
        status, ratio, sumber = processors.resolve_sumber_dataset(
            BASE_DIR, row["Sensor"], row["Model"]
        )
        status_list.append(status)
        rasio_list.append(ratio)
        sumber_list.append(sumber)

    df["Kondisi_Awal_Data"] = status_list
    df["Rasio_Kelas_Asli"] = rasio_list
    df["Sumber_Dataset"] = sumber_list

    df["Mekanisme_Balancing"] = df.apply(
        lambda row: tentukan_mekanisme_balancing(row["Model"]),
        axis=1,
    )

    # Perangkingan Standar Akademis Murni: Diurutkan Berdasarkan F1-Score -> Accuracy
    df_sorted = df.sort_values(by=["F1_Score", "Accuracy"], ascending=False).reset_index(drop=True)

    # Format Persentase untuk Tampilan Tabel Akademis
    df_md = df_sorted.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1_Score"]:
        if col in df_md.columns:
            df_md[col] = df_md[col].apply(lambda x: f"{x * 100:.2f}%" if x <= 1.0 else f"{x:.2f}%")

    kolom_cetak = [
        "Model",
        "Sensor",
        "Sumber_Dataset",
        "Kondisi_Awal_Data",
        "Rasio_Kelas_Asli",
        "Mekanisme_Balancing",
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score",
    ]
    df_md_cetak = df_md[kolom_cetak].copy()

    # Menggunakan pustaka tabulate untuk format tabel Markdown standar jurnal/skripsi
    tabel_markdown = tabulate(df_md_cetak, headers="keys", tablefmt="pipe", showindex=False)

    with open(md_output, "w", encoding="utf-8") as f:
        f.write("# 📊 PAPAN PERINGKAT KOMPREHENSIF EVALUASI MODEL HRV SKRIPSI\n\n")
        f.write(
            "Berikut adalah tabel ringkasan komparasi performa kuantitatif model komprehensif. "
            "Pengurutan peringkat menggunakan standar *F1-Score First* untuk mengakomodasi "
            "kondisi ketidakseimbangan dataset (*imbalanced data*).\n\n"
        )
        f.write(tabel_markdown)
        f.write("\n")

    print(f"SUCCESS: Berkas rekap komprehensif V6 berhasil diterbitkan di: {md_output}")


if __name__ == "__main__":
    main()
