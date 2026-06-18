"""
Fase 07: Sistem Rekapitulasi Eksperimen Komprehensif
Deskripsi: Memuat database master hasil pengujian seluruh variasi model,
           melakukan sorting berbasis metrik F1-Score untuk akurasi yang objektif,
           serta mengekspor tabel ringkasan ke dalam format cetak akademis.
"""

import os
import pandas as pd

# Konfigurasi Jalur Direktori
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "01_processed", "REKAP_AKURASI_MASTER.csv")

def catat_hasil_eksperimen(nama_model, sensor, akurasi, presisi, recall, f1_score):
    """
    Fungsi global untuk mencatat hasil pengujian model secara live dari notebook manapun.
    Fungsi ini otomatis membuat file CSV baru jika belum eksis.
    """
    data_baru = pd.DataFrame([{
        "Nama_Model": nama_model, "Sensor": sensor,
        "Akurasi_Persen": round(akurasi, 2), "Precision_Persen": round(presisi, 2),
        "Recall_Persen": round(recall, 2), "F1_Score_Persen": round(f1_score, 2)
    }])
    
    if not os.path.exists(CSV_PATH):
        data_baru.to_csv(CSV_PATH, index=False)
    else:
        data_baru.to_csv(CSV_PATH, mode='a', header=False, index=False)

def main():
    print("LOG: [Membangun Tabel Ringkasan Eksperimen Skripsi]")
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: Berkas database master {CSV_PATH} belum terbentuk. Jalankan pipeline modeling terlebih dahulu.")
        return
        
    df = pd.read_csv(CSV_PATH)
    
    # Prioritaskan pengurutan berdasarkan F1-Score untuk mengantisipasi ketidakseimbangan data target
    df_sorted = df.sort_values(by=["F1_Score_Persen", "Akurasi_Persen"], ascending=False)
    
    print("\n" + "="*85)
    print("🏆 PAPAN PERINGKAT PERFORMA MODEL DETEKSI BEBAN KOGNITIF (SORTED BY F1-SCORE) 🏆")
    print("="*85)
    print(df_sorted.to_string(index=False))
    print("="*85)
    
    # Ekspor ke file Markdown agar bisa langsung dicopy-paste ke draf skripsi Anda
    md_path = os.path.join(BASE_DIR, "data", "01_processed", "TABEL_RINGKASAN_SKRIPSI.md")
    df_sorted.to_markdown(md_path, index=False)
    print(f"LOG: Berhasil mengekspor tabel format Markdown ke {md_path}")

if __name__ == "__main__":
    main()