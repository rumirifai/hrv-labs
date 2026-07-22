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

# Baseline DL CM
cm = np.array([[1290, 1546], [1149, 1951]])

fig, ax = plt.subplots(figsize=(5.5, 4.5))
fig.patch.set_facecolor("white")
labels = ["Low", "High"]

sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", cbar=True,
            xticklabels=labels, yticklabels=labels, ax=ax,
            annot_kws={"size": 13, "weight": "bold"})
ax.set_title("Confusion Matrix CNN-LSTM Baseline", pad=12, fontweight="bold")
ax.set_xlabel("Predicted Label", labelpad=8)
ax.set_ylabel("True Label", labelpad=8)

plt.tight_layout()

# ─── Simpan ───────────────────────────────────────────────────────────────────
FILE_NAME  = "Gambar 4.9 Confusion Matrix CNN-LSTM Baseline (Waveform V2).png"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SCRIPT_DIR, "result")
os.makedirs(RESULT_DIR, exist_ok=True)

for dest in [os.path.join(SCRIPT_DIR, FILE_NAME), os.path.join(RESULT_DIR, FILE_NAME)]:
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[OK] Tersimpan: {dest}")
print("[OK] Selesai me-render Gambar 4.9")
