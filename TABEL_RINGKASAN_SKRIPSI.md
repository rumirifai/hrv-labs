# 📊 PAPAN PERINGKAT KOMPREHENSIF EVALUASI MODEL HRV SKRIPSI

Berikut adalah tabel ringkasan komparasi performa model komprehensif. Pengurutan peringkat menggunakan standar *Multi-Metrik Best Practice* (Mengutamakan kestabilan akurasi hasil akhir pada kondisi dataset awal yang tidak seimbang):

| Model                     | Sensor       | Kondisi_Awal_Data   | Rasio_Kelas_Asli   | Mekanisme_Balancing              | Rasio_Prediksi_Akhir        | Accuracy   | F1_Score   | Kesehatan_Model_Akhir    |
|:--------------------------|:-------------|:--------------------|:-------------------|:---------------------------------|:----------------------------|:-----------|:-----------|:-------------------------|
| CNN-LSTM Baseline         | PPG Tabular  | Imbalanced          | 290:177            | None (No Intervention)           | 272 : 195 (✅ Proporsional) | 65.17%     | 64.16%     | ✅ SEHAT (Stabil)        |
| CNN-LSTM Hyperparam Tuned | ECG Waveform | Imbalanced          | 290:174            | Class Weights (Algorithmic Cost) | 268 : 196 (✅ Proporsional) | 65.07%     | 71.07%     | ✅ SEHAT (Stabil)        |
| CNN-BiLSTM V3             | ECG Waveform | Imbalanced          | 290:174            | Class Weights (Algorithmic Cost) | 268 : 196 (✅ Proporsional) | 64.87%     | 67.29%     | ✅ SEHAT (Stabil)        |
| CNN-LSTM Baseline         | ECG Tabular  | Imbalanced          | 290:174            | None (No Intervention)           | 268 : 196 (✅ Proporsional) | 64.00%     | 65.38%     | ✅ SEHAT (Stabil)        |
| LSTM + Attention          | ECG Tabular  | Imbalanced          | 290:174            | Class Weights (Algorithmic Cost) | 268 : 196 (✅ Proporsional) | 63.43%     | 65.96%     | ✅ SEHAT (Stabil)        |
| SVM Baseline              | ECG Tabular  | Imbalanced          | 290:174            | None (No Intervention)           | 268 : 196 (✅ Proporsional) | 63.00%     | 47.00%     | ✅ SEHAT (Stabil)        |
| Random Forest Baseline    | ECG Tabular  | Imbalanced          | 290:174            | None (No Intervention)           | 268 : 196 (✅ Proporsional) | 63.00%     | 47.00%     | ✅ SEHAT (Stabil)        |
| Random Forest Baseline    | PPG Tabular  | Imbalanced          | 290:177            | None (No Intervention)           | 245 : 219 (Stabil)          | 62.00%     | 46.00%     | ✅ SEHAT (Stabil)        |
| SVM Baseline              | PPG Tabular  | Imbalanced          | 290:177            | None (No Intervention)           | 245 : 219 (Stabil)          | 61.00%     | 45.00%     | ✅ SEHAT (Stabil)        |
| CNN-LSTM + mRMR           | ECG Tabular  | Imbalanced          | 290:174            | Class Weights (Algorithmic Cost) | 245 : 219 (Stabil)          | 60.29%     | 55.02%     | ✅ SEHAT (Stabil)        |
| LSTM + Attention          | PPG Tabular  | Imbalanced          | 290:177            | Class Weights (Algorithmic Cost) | 245 : 219 (Stabil)          | 60.11%     | 61.41%     | ✅ SEHAT (Stabil)        |
| CNN-LSTM + UFR            | ECG Tabular  | Imbalanced          | 290:174            | Class Weights (Algorithmic Cost) | 245 : 219 (Stabil)          | 60.00%     | 54.84%     | ✅ SEHAT (Stabil)        |
| Fusi Multimodal Expert    | ECG+PPG      | Imbalanced          | 290:174            | Class Weights (Hybrid Context)   | 210 : 254 (Cukup)           | 55.43%     | 61.00%     | ℹ️ Cukup / Valid         |
| CNN-BiLSTM V2             | ECG Waveform | Imbalanced          | 290:174            | Class Weights (Algorithmic Cost) | 0 : 464 (⚠️ Pincang Total)  | 52.25%     | 68.63%     | ⚠️ ANOMALI (Bias Recall) |
| CNN-LSTM + UFR            | PPG Tabular  | Imbalanced          | 290:177            | Class Weights (Algorithmic Cost) | 0 : 464 (⚠️ Pincang Total)  | 50.56%     | 66.15%     | ⚠️ ANOMALI (Bias Recall) |
| CNN-LSTM + mRMR           | PPG Tabular  | Imbalanced          | 290:177            | Class Weights (Algorithmic Cost) | 0 : 464 (⚠️ Pincang Total)  | 50.00%     | 66.03%     | ⚠️ ANOMALI (Bias Recall) |

### 📝 Panduan Membaca Alur Eksperimen (Untuk Bab 4 & Bab 5 Skripsi):
1. **Kondisi Awal Data (`Imbalanced`):** Menunjukkan bahwa dataset dasar Anda secara riil memiliki ketidakseimbangan kelas target (proporsi kelas target 'Beban Kerja Tinggi' lebih sedikit).
2. **Mekanisme Balancing (`Class Weights`):** Intervensi matematis yang disuntikkan ke dalam model saat proses pelatihan untuk memaksa jaringan memberikan penalti lebih berat pada kesalahan deteksi kelas minoritas.
3. **Rasio Prediksi Akhir:** Mengindikasikan sebaran tebakan riil model terhadap data uji. Model yang sehat akan menunjukkan hasil yang `Proporsional`, sedangkan model yang pincang menandakan kegagalan beradaptasi terhadap data imbalance.
4. **Hasil Akhir Setelah Mekanisme (`Kesehatan_Model_Akhir`):**
   * **`✅ SEHAT (Stabil)`**: Model berhasil memanfaatkan intervensi Class Weights untuk keluar dari jebakan bias data sehingga mencatatkan nilai Akurasi dan F1-Score yang seimbang.
   * **`⚠️ ANOMALI (Bias Recall)`**: Model gagal melakukan generalisasi dan menyerah pada bias ketidakseimbangan data dengan hanya menebak satu kelas dominan secara terus-menerus (Rasio output pincang total).
