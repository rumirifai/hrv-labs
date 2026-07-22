# 📊 PAPAN PERINGKAT KOMPREHENSIF EVALUASI MODEL HRV SKRIPSI

Berikut adalah tabel ringkasan komparasi performa model komprehensif. Pengurutan peringkat menggunakan standar *Multi-Metrik Best Practice* (Mengutamakan kestabilan akurasi hasil akhir pada kondisi dataset awal yang tidak seimbang). Distribusi kelas dihitung dari berkas CSV/NPY riil; rasio prediksi diambil dari inferensi model `.keras` bila kompatibel, atau diestimasi dari metrik evaluasi pada dataset audit yang sama.

| Model | Sensor | Sumber_Dataset | Kondisi_Awal_Data | Rasio_Kelas_Asli | Mekanisme_Balancing | Rasio_Prediksi_Akhir | Accuracy | F1_Score | Kesehatan_Model_Akhir |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CNN-LSTM Baseline | PPG Tabular | PPG Tabular | Imbalanced | 290:177 | Class Weights (Algorithmic Cost) | 187 : 169 (✅ Proporsional) [Audit Model] | 65.17% | 64.16% | ✅ SEHAT (Stabil) |
| CNN-LSTM Hyperparam Tuned | ECG Waveform | ECG Waveform V2 | Balanced | 14177:15501 | Class Weights (Algorithmic Cost) | 9350 : 20328 (✅ Proporsional) [Audit Model] | 65.07% | 71.07% | ✅ SEHAT (Stabil) |
| CNN-BiLSTM V3 | ECG Waveform | ECG Waveform V3 | Balanced | 6528:7629 | Class Weights (Algorithmic Cost) | 6583 : 7574 (✅ Proporsional) [Audit Model] | 64.87% | 67.29% | ✅ SEHAT (Stabil) |
| CNN-LSTM Baseline | ECG Tabular | ECG Tabular | Imbalanced | 290:174 | Class Weights (Algorithmic Cost) | 158 : 192 (✅ Proporsional) [Audit Model] | 64.00% | 65.38% | ✅ SEHAT (Stabil) |
| LSTM + Attention | ECG Tabular | ECG Tabular | Imbalanced | 290:174 | Class Weights (Algorithmic Cost) | 146 : 204 (✅ Proporsional) [Audit Model] | 63.43% | 65.96% | ✅ SEHAT (Stabil) |
| SVM Baseline | ECG Tabular | ECG Tabular | Imbalanced | 290:174 | Class Weights (Sklearn Balanced) | N/A (Model Klasik - Tidak Diaudit) | 63.00% | 47.00% | ✅ SEHAT (Stabil) |
| Random Forest Baseline | ECG Tabular | ECG Tabular | Imbalanced | 290:174 | Class Weights (Sklearn Balanced) | N/A (Model Klasik - Tidak Diaudit) | 63.00% | 47.00% | ✅ SEHAT (Stabil) |
| Random Forest Baseline | PPG Tabular | PPG Tabular | Imbalanced | 290:177 | Class Weights (Sklearn Balanced) | N/A (Model Klasik - Tidak Diaudit) | 62.00% | 46.00% | ✅ SEHAT (Stabil) |
| SVM Baseline | PPG Tabular | PPG Tabular | Imbalanced | 290:177 | Class Weights (Sklearn Balanced) | N/A (Model Klasik - Tidak Diaudit) | 61.00% | 45.00% | ✅ SEHAT (Stabil) |
| CNN-LSTM + mRMR | ECG Tabular | ECG Tabular | Imbalanced | 290:174 | Class Weights + Feature Weighting (UFR/mRMR) | 213 : 137 (✅ Proporsional) [Audit Model] | 60.29% | 55.02% | ✅ SEHAT (Stabil) |
| LSTM + Attention | PPG Tabular | PPG Tabular | Imbalanced | 290:177 | Class Weights (Algorithmic Cost) | 165 : 191 (✅ Proporsional) [Audit Model] | 60.11% | 61.41% | ✅ SEHAT (Stabil) |
| CNN-LSTM + UFR | ECG Tabular | ECG Tabular | Imbalanced | 290:174 | Class Weights + Feature Weighting (UFR/mRMR) | 212 : 138 (✅ Proporsional) [Audit Model] | 60.00% | 54.84% | ✅ SEHAT (Stabil) |
| Fusi Multimodal Expert | ECG+PPG | Multimodal (Inner Join ECG+PPG) | Imbalanced | 978:174 | Class Weights (Hybrid Context) | 122 : 228 (Stabil) [Audit Model] | 55.43% | 61.00% | ℹ️ Cukup / Valid |
| CNN-BiLSTM V2 | ECG Waveform | ECG Waveform V2 | Balanced | 14177:15501 | Class Weights (Algorithmic Cost) | 7 : 29671 (⚠️ Pincang Total) [Audit Model] | 52.25% | 68.63% | ⚠️ ANOMALI (Bias Recall) |
| CNN-LSTM + UFR | PPG Tabular | PPG Tabular | Imbalanced | 290:177 | Class Weights + Feature Weighting (UFR/mRMR) | 13 : 343 (⚠️ Pincang Total) [Audit Model] | 50.56% | 66.15% | ⚠️ ANOMALI (Bias Recall) |
| CNN-LSTM + mRMR | PPG Tabular | PPG Tabular | Imbalanced | 290:177 | Class Weights + Feature Weighting (UFR/mRMR) | 9 : 347 (⚠️ Pincang Total) [Audit Model] | 50.00% | 66.03% | ⚠️ ANOMALI (Bias Recall) |

### 📝 Panduan Membaca Alur Eksperimen (Untuk Bab 4 & Bab 5 Skripsi):
1. **Sumber Dataset:** Menunjukkan berkas aktual yang menjadi acuan audit (tabular CSV Fase 01 atau waveform NPY Fase 02).
2. **Kondisi Awal Data (`Imbalanced`):** Dihitung dari distribusi kelas riil pada sumber data. Status `Balanced` diberikan jika proporsi kelas minoritas >= 40%.
3. **Mekanisme Balancing:** Mencerminkan intervensi yang benar-benar dipakai pada kode eksperimen (Sklearn balanced, class weights Keras, atau feature weighting).
4. **Rasio Prediksi Akhir:** Diambil langsung dari inferensi model `.keras` jika kompatibel, atau diestimasi dari metrik Accuracy/Precision/Recall pada dataset audit yang sama. Model klasik (SVM/RF) ditandai N/A.
5. **Hasil Akhir Setelah Mekanisme (`Kesehatan_Model_Akhir`):**
   * **`✅ SEHAT (Stabil)`**: Akurasi >= 60% dengan distribusi prediksi proporsional.
   * **`⚠️ ANOMALI (Bias Recall)`**: Recall sangat tinggi (>= 90%) dengan akurasi rendah (<= 55%), atau prediksi hanya satu kelas.
