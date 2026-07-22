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

# SVM CM
cm_ecg = np.array([[94, 84], [62, 110]])
cm_ppg = np.array([[103, 76], [69, 108]])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.patch.set_facecolor("white")
labels = ["Low", "High"]

# ECG
sns.heatmap(cm_ecg, annot=True, fmt="d", cmap="Blues", cbar=True,
            xticklabels=labels, yticklabels=labels, ax=axes[0],
            annot_kws={"size": 13, "weight": "bold"})
axes[0].set_title("4.7 (a)  SVM Tabular - Sensor ECG", pad=10, fontweight="bold")
axes[0].set_xlabel("Predicted Label", labelpad=6)
axes[0].set_ylabel("True Label", labelpad=6)

# PPG
sns.heatmap(cm_ppg, annot=True, fmt="d", cmap="Blues", cbar=True,
            xticklabels=labels, yticklabels=labels, ax=axes[1],
            annot_kws={"size": 13, "weight": "bold"})
axes[1].set_title("4.7 (b)  SVM Tabular - Sensor PPG", pad=10, fontweight="bold")
axes[1].set_xlabel("Predicted Label", labelpad=6)
axes[1].set_ylabel("True Label", labelpad=6)

plt.tight_layout()

# ─── Simpan ───────────────────────────────────────────────────────────────────
FILE_NAME  = "Gambar 4.7 Confusion Matrix Model Klasik SVM Tabular.png"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SCRIPT_DIR, "result")
os.makedirs(RESULT_DIR, exist_ok=True)

for dest in [os.path.join(SCRIPT_DIR, FILE_NAME), os.path.join(RESULT_DIR, FILE_NAME)]:
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[OK] Tersimpan: {dest}")
print("[OK] Selesai me-render Gambar 4.7")
