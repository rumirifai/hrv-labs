# 📊 PAPAN PERINGKAT KOMPREHENSIF EVALUASI MODEL HRV SKRIPSI

Berikut adalah tabel ringkasan komparasi performa kuantitatif model komprehensif. Pengurutan peringkat menggunakan standar *F1-Score First* untuk mengakomodasi kondisi ketidakseimbangan dataset (*imbalanced data*).

| Model                     | Sensor       | Sumber_Dataset                  | Kondisi_Awal_Data   | Rasio_Kelas_Asli   | Mekanisme_Balancing                         | Accuracy   | Precision   | Recall   | F1_Score   |
|:--------------------------|:-------------|:--------------------------------|:--------------------|:-------------------|:--------------------------------------------|:-----------|:------------|:---------|:-----------|
| CNN-LSTM Hyperparam Tuned | ECG Waveform | ECG Waveform V2                 | Balanced            | 14177:15501        | Tanpa Intervensi / Grid Search (Unweighted) | 65.07%     | 62.63%      | 82.13%   | 71.07%     |
| CNN-BiLSTM V2             | ECG Waveform | ECG Waveform V2                 | Balanced            | 14177:15501        | Class Weights (Algorithmic Cost)            | 52.25%     | 52.24%      | 100.00%  | 68.63%     |
| CNN-BiLSTM V3             | ECG Waveform | ECG Waveform V3                 | Balanced            | 6528:7629          | Class Weights + Cosine LR Scheduler         | 64.87%     | 67.53%      | 67.05%   | 67.29%     |
| CNN-LSTM + UFR            | PPG Tabular  | PPG Tabular                     | Imbalanced          | 290:177            | Class Weights + Feature Weighting (UFR)     | 50.56%     | 50.15%      | 97.18%   | 66.15%     |
| CNN-LSTM + mRMR           | PPG Tabular  | PPG Tabular                     | Imbalanced          | 290:177            | Class Weights + Feature Weighting (mRMR)    | 50.00%     | 49.86%      | 97.74%   | 66.03%     |
| LSTM + Attention          | ECG Tabular  | ECG Tabular                     | Imbalanced          | 290:174            | Class Weights (Algorithmic Cost)            | 63.43%     | 60.78%      | 72.09%   | 65.96%     |
| CNN-LSTM Baseline         | ECG Tabular  | ECG Tabular                     | Imbalanced          | 290:174            | Class Weights (Algorithmic Cost)            | 64.00%     | 61.98%      | 69.19%   | 65.38%     |
| CNN-LSTM Baseline         | PPG Tabular  | PPG Tabular                     | Imbalanced          | 290:177            | Class Weights (Algorithmic Cost)            | 65.17%     | 65.68%      | 62.71%   | 64.16%     |
| LSTM + Attention          | PPG Tabular  | PPG Tabular                     | Imbalanced          | 290:177            | Class Weights (Algorithmic Cost)            | 60.11%     | 59.16%      | 63.84%   | 61.41%     |
| Fusi Multimodal Expert    | ECG+PPG      | Multimodal (Inner Join ECG+PPG) | Imbalanced          | 978:174            | Class Weights (Hybrid Context)              | 55.43%     | 53.51%      | 70.93%   | 61.00%     |
| Random Forest Baseline    | PPG Tabular  | PPG Tabular                     | Imbalanced          | 290:177            | Class Weights (Sklearn Balanced)            | 60.41%     | 59.82%      | 62.71%   | 60.81%     |
| SVM Baseline              | PPG Tabular  | PPG Tabular                     | Imbalanced          | 290:177            | Class Weights (Sklearn Balanced)            | 59.29%     | 59.76%      | 60.98%   | 59.48%     |
| SVM Baseline              | ECG Tabular  | ECG Tabular                     | Imbalanced          | 290:174            | Class Weights (Sklearn Balanced)            | 58.29%     | 75.00%      | 63.95%   | 59.46%     |
| Random Forest Baseline    | ECG Tabular  | ECG Tabular                     | Imbalanced          | 290:174            | Class Weights (Sklearn Balanced)            | 58.86%     | 57.64%      | 57.94%   | 57.40%     |
| CNN-LSTM + mRMR           | ECG Tabular  | ECG Tabular                     | Imbalanced          | 290:174            | Class Weights + Feature Weighting (mRMR)    | 60.29%     | 62.04%      | 49.42%   | 55.02%     |
| CNN-LSTM + UFR            | ECG Tabular  | ECG Tabular                     | Imbalanced          | 290:174            | Class Weights + Feature Weighting (UFR)     | 60.00%     | 61.59%      | 49.42%   | 54.84%     |
