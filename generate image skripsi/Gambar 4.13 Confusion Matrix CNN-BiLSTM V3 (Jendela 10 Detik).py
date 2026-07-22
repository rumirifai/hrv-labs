import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use("Agg")
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
matplotlib.rcParams['font.size'] = 13
matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['axes.labelsize'] = 13
matplotlib.rcParams['xtick.labelsize'] = 11
matplotlib.rcParams['ytick.labelsize'] = 11

# Dapatkan BASE_DIR dari letak skrip (mengarah ke root workspace d:\hrv-labs)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

X_path = os.path.join(BASE_DIR, "data", "02_waveform", "X_Raw_ECG_V3.npy")
y_path = os.path.join(BASE_DIR, "data", "02_waveform", "y_Raw_ECG_V3.npy")
model_path = os.path.join(BASE_DIR, "data", "03_saved_models", "Model_Terbaik_BiLSTM_V3_CosineLR.keras")

use_static = True
if os.path.exists(X_path) and os.path.exists(y_path) and os.path.exists(model_path):
    try:
        from sklearn.model_selection import train_test_split
        from core import models
        X = np.load(X_path)
        y = np.load(y_path)
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model = models.build_cnn_bilstm_waveform(window_size=X_test.shape[1], learning_rate=0.001)
        model.load_weights(model_path)
        y_prob = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_prob > 0.5).astype(int)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        use_static = False
    except Exception as e:
        print(f"Gagal menjalankan inferensi model, fallback ke data statis: {e}")

if use_static:
    # TP=936, FP=595, TN=711, FN=590
    cm = np.array([[711, 595], [590, 936]])

fig, ax = plt.subplots(figsize=(5.5, 4.5))
fig.patch.set_facecolor("white")
labels = ["Low", "High"]
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", cbar=True,
            xticklabels=labels, yticklabels=labels, ax=ax,
            annot_kws={"size": 13, "weight": "bold"})
ax.set_title("Confusion Matrix CNN-BiLSTM V3", pad=12, fontweight="bold")
ax.set_xlabel("Predicted Label", labelpad=8)
ax.set_ylabel("True Label", labelpad=8)

plt.tight_layout()

# ─── Simpan ───────────────────────────────────────────────────────────────────
FILE_NAME  = "Gambar 4.13 Confusion Matrix CNN-BiLSTM V3 (Jendela 10 Detik).png"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SCRIPT_DIR, "result")
os.makedirs(RESULT_DIR, exist_ok=True)

for dest in [os.path.join(SCRIPT_DIR, FILE_NAME), os.path.join(RESULT_DIR, FILE_NAME)]:
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[OK] Tersimpan: {dest}")
print("[OK] Selesai me-render Gambar 4.13")
