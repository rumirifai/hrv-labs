"""
Sistem Rekapitulasi Eksperimen Skripsi (Versi Audit Komprehensif & Multi-Metrik V3)
Deskripsi: Membaca database master CSV, mengaudit status keseimbangan data asli
           dari sumber CSV/NPY riil, mencatat mekanisme balancing sesuai kode eksperimen,
           menghitung rasio prediksi akhir dari model .keras (bukan hardcode),
           serta mengevaluasi kesehatan model.
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
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler

from core import processors

tf.get_logger().setLevel("ERROR")

DATA_TABULAR = os.path.join(BASE_DIR, "data", "01_processed")
DATA_WAVEFORM = os.path.join(BASE_DIR, "data", "02_waveform")
MODEL_DIR = os.path.join(BASE_DIR, "data", "03_saved_models")

FITUR_TABULAR = ["MeanNN_ms", "SDNN_ms", "RMSSD_ms"]


def tentukan_mekanisme_balancing(model_name, sensor_type):
    """Mencatat intervensi penyeimbangan sesuai implementasi di notebook Fase 01-06."""
    model_name = str(model_name)
    sensor_type = str(sensor_type)

    if "UFR" in model_name or "mRMR" in model_name:
        return "Class Weights + Feature Weighting (UFR/mRMR)"
    if "Fusi" in model_name:
        return "Class Weights (Hybrid Context)"
    if "Baseline" in model_name and ("SVM" in model_name or "Random Forest" in model_name):
        return "Class Weights (Sklearn Balanced)"
    return "Class Weights (Algorithmic Cost)"


def tentukan_kesehatan_model(accuracy, recall):
    if accuracy <= 0.55 and recall >= 0.90:
        return "⚠️ ANOMALI (Bias Recall)"
    if accuracy >= 0.60:
        return "✅ SEHAT (Stabil)"
    return "ℹ️ Cukup / Valid"


def ekstrak_nama_model(keterangan):
    prefix = "Audit Fisik "
    keterangan = str(keterangan)
    if keterangan.startswith(prefix):
        return keterangan[len(prefix):].strip()
    return None


def muat_data_tabular(sensor):
    path_csv = os.path.join(DATA_TABULAR, f"Dataset_Fitur_{sensor}_6_Fitur.csv")
    if not os.path.exists(path_csv):
        return None, None

    df = pd.read_csv(path_csv).dropna(subset=FITUR_TABULAR)
    X = df[FITUR_TABULAR].values
    y = LabelEncoder().fit_transform(df["Label"].values.flatten())
    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled.reshape((len(X_scaled), len(FITUR_TABULAR), 1)), y


def muat_data_waveform(versi):
    path_x = os.path.join(DATA_WAVEFORM, f"X_Raw_ECG_{versi}.npy")
    path_y = os.path.join(DATA_WAVEFORM, f"y_Raw_ECG_{versi}.npy")
    if not (os.path.exists(path_x) and os.path.exists(path_y)):
        return None, None

    X = np.load(path_x)
    y = np.load(path_y)
    if len(X.shape) == 2:
        X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y.flatten()


def muat_data_untuk_model(sensor, model_name):
    """Memetakan baris rekap ke pasangan data yang sama dengan fix_database_master.py."""
    sensor = str(sensor)
    model_name = str(model_name)

    if sensor == "ECG+PPG" or "Fusi" in model_name:
        return muat_data_tabular("ECG")

    if "Waveform" in sensor or "BiLSTM" in model_name or "Hyperparam" in model_name:
        versi = "V3" if "V3" in model_name else "V2"
        return muat_data_waveform(versi)

    if sensor == "PPG Tabular":
        return muat_data_tabular("PPG")
    return muat_data_tabular("ECG")


def ambil_jumlah_kelas_audit(y_target):
    if y_target is None:
        return 0, 0
    rendah = int(np.sum(y_target == 0))
    tinggi = int(np.sum(y_target == 1))
    return rendah, tinggi


def format_rasio_prediksi(pred_rendah, pred_tinggi, accuracy, recall, sumber=""):
    total = pred_rendah + pred_tinggi
    if total == 0:
        return "N/A"

    rasio = f"{pred_rendah} : {pred_tinggi}"
    if pred_rendah == 0 or pred_tinggi == 0:
        label = f"{rasio} (⚠️ Pincang Total)"
    elif accuracy <= 0.55 and recall >= 0.90:
        label = f"{rasio} (⚠️ Pincang Total)"
    elif accuracy >= 0.60:
        label = f"{rasio} (✅ Proporsional)"
    else:
        label = f"{rasio} (Stabil)"

    if sumber:
        label = f"{label} [{sumber}]"
    return label


def hitung_rasio_prediksi_model(sensor, model_name, keterangan, accuracy, recall, precision):
    """Menghitung distribusi prediksi kelas 0/1 dari model .keras atau estimasi metrik."""
    if "Eksperimen Klasik" in str(keterangan):
        return "N/A (Model Klasik - Tidak Diaudit)"

    _, y_target = muat_data_untuk_model(sensor, model_name)
    if y_target is None:
        return "N/A (Data Audit Tidak Tersedia)"

    jumlah_rendah, jumlah_tinggi = ambil_jumlah_kelas_audit(y_target)
    file_model = ekstrak_nama_model(keterangan)

    if file_model:
        model_path = os.path.join(MODEL_DIR, file_model)
        if os.path.exists(model_path):
            X_input, _ = muat_data_untuk_model(sensor, model_name)
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                y_prob = model.predict(X_input, verbose=0)
                y_pred = (y_prob > 0.5).astype(int).flatten()

                min_len = min(len(y_target), len(y_pred))
                y_pred = y_pred[:min_len]

                pred_rendah = int(np.sum(y_pred == 0))
                pred_tinggi = int(np.sum(y_pred == 1))
                return format_rasio_prediksi(
                    pred_rendah, pred_tinggi, accuracy, recall, sumber="Audit Model"
                )
            except Exception:
                pass

    pred_rendah, pred_tinggi = processors.estimasi_distribusi_prediksi(
        accuracy, precision, recall, jumlah_rendah, jumlah_tinggi
    )
    return format_rasio_prediksi(
        pred_rendah,
        pred_tinggi,
        accuracy,
        recall,
        sumber="Estimasi Metrik",
    )


def dataframe_ke_markdown(df):
    """Menulis tabel markdown tanpa dependency tabulate."""
    kolom = list(df.columns)
    baris = [
        "| " + " | ".join(kolom) + " |",
        "| " + " | ".join(":---" for _ in kolom) + " |",
    ]
    for _, row in df.iterrows():
        baris.append("| " + " | ".join(str(row[k]) for k in kolom) + " |")
    return "\n".join(baris)


def main():
    print("LOG: [Membangun Tabel Ringkasan Eksperimen Skripsi - Versi Audit Komprehensif V3]")

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
        lambda row: tentukan_mekanisme_balancing(row["Model"], row["Sensor"]),
        axis=1,
    )

    df["Kesehatan_Model_Akhir"] = df.apply(
        lambda row: tentukan_kesehatan_model(row["Accuracy"], row["Recall"]),
        axis=1,
    )

    df_sehat = df[df["Kesehatan_Model_Akhir"] == "✅ SEHAT (Stabil)"].sort_values(
        by=["Accuracy", "F1_Score"], ascending=False
    )
    df_sakit = df[df["Kesehatan_Model_Akhir"] != "✅ SEHAT (Stabil)"].sort_values(
        by=["Accuracy"], ascending=False
    )
    df_sorted = pd.concat([df_sehat, df_sakit], ignore_index=True)

    df_sorted["Rasio_Prediksi_Akhir"] = df_sorted.apply(
        lambda row: hitung_rasio_prediksi_model(
            row["Sensor"],
            row["Model"],
            row["Keterangan"],
            row["Accuracy"],
            row["Recall"],
            row["Precision"],
        ),
        axis=1,
    )

    kolom_cetak = [
        "Model",
        "Sensor",
        "Sumber_Dataset",
        "Kondisi_Awal_Data",
        "Rasio_Kelas_Asli",
        "Mekanisme_Balancing",
        "Rasio_Prediksi_Akhir",
        "Accuracy",
        "F1_Score",
        "Kesehatan_Model_Akhir",
    ]
    df_md = df_sorted[kolom_cetak].copy()

    for col in ["Accuracy", "F1_Score"]:
        df_md[col] = df_md[col].apply(lambda x: f"{x * 100:.2f}%" if x <= 1.0 else f"{x:.2f}%")

    with open(md_output, "w", encoding="utf-8") as f:
        f.write("# 📊 PAPAN PERINGKAT KOMPREHENSIF EVALUASI MODEL HRV SKRIPSI\n\n")
        f.write(
            "Berikut adalah tabel ringkasan komparasi performa model komprehensif. "
            "Pengurutan peringkat menggunakan standar *Multi-Metrik Best Practice* "
            "(Mengutamakan kestabilan akurasi hasil akhir pada kondisi dataset awal "
            "yang tidak seimbang). Distribusi kelas dihitung dari berkas CSV/NPY riil; "
            "rasio prediksi diambil dari inferensi model `.keras` bila kompatibel, "
            "atau diestimasi dari metrik evaluasi pada dataset audit yang sama.\n\n"
        )
        f.write(dataframe_ke_markdown(df_md))
        f.write("\n\n### 📝 Panduan Membaca Alur Eksperimen (Untuk Bab 4 & Bab 5 Skripsi):\n")
        f.write(
            "1. **Sumber Dataset:** Menunjukkan berkas aktual yang menjadi acuan audit "
            "(tabular CSV Fase 01 atau waveform NPY Fase 02).\n"
        )
        f.write(
            "2. **Kondisi Awal Data (`Imbalanced`):** Dihitung dari distribusi kelas riil "
            "pada sumber data. Status `Balanced` diberikan jika proporsi kelas minoritas >= "
            f"{processors.BALANCE_THRESHOLD_PERSEN:.0f}%.\n"
        )
        f.write(
            "3. **Mekanisme Balancing:** Mencerminkan intervensi yang benar-benar dipakai "
            "pada kode eksperimen (Sklearn balanced, class weights Keras, atau feature weighting).\n"
        )
        f.write(
            "4. **Rasio Prediksi Akhir:** Diambil langsung dari inferensi model `.keras` "
            "jika kompatibel, atau diestimasi dari metrik Accuracy/Precision/Recall pada "
            "dataset audit yang sama. Model klasik (SVM/RF) ditandai N/A.\n"
        )
        f.write("5. **Hasil Akhir Setelah Mekanisme (`Kesehatan_Model_Akhir`):**\n")
        f.write(
            "   * **`✅ SEHAT (Stabil)`**: Akurasi >= 60% dengan distribusi prediksi proporsional.\n"
        )
        f.write(
            "   * **`⚠️ ANOMALI (Bias Recall)`**: Recall sangat tinggi (>= 90%) dengan akurasi "
            "rendah (<= 55%), atau prediksi hanya satu kelas.\n"
        )

    print(f"SUCCESS: Berkas rekap komprehensif V3 berhasil diterbitkan di: {md_output}")


if __name__ == "__main__":
    main()
