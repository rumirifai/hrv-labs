import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 11
matplotlib.rcParams['ytick.labelsize'] = 11
matplotlib.rcParams['legend.fontsize'] = 11

# Dapatkan BASE_DIR dari letak skrip (mengarah ke root workspace d:\hrv-labs)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

features = [
    "MeanNN\n(ms)",
    "SDNN\n(ms)",
    "RMSSD\n(ms)",
    "LF Power\n(nu)",
    "HF Power\n(nu)",
    "LF/HF\nRatio"
]

# Korelasi Pearson ECG asli dari Gambar 4.20 Skripsi
corr_ecg = np.array([
    [1.00, 0.36, 0.35, -0.11, -0.19, 0.14],
    [0.36, 1.00, 0.98, -0.30, -0.32, 0.30],
    [0.35, 0.98, 1.00, -0.30, -0.30, 0.27],
    [-0.11, -0.30, -0.30, 1.00, 0.51, -0.04],
    [-0.19, -0.32, -0.30, 0.51, 1.00, -0.67],
    [0.14, 0.30, 0.27, -0.04, -0.67, 1.00]
])

# Korelasi Pearson PPG asli dari Gambar 4.20 Skripsi
corr_ppg = np.array([
    [1.00, 0.11, 0.11, -0.08, -0.17, 0.13],
    [0.11, 1.00, 0.94, -0.18, 0.50, -0.35],
    [0.11, 0.94, 1.00, -0.04, 0.65, -0.55],
    [-0.08, -0.18, -0.04, 1.00, 0.32, -0.08],
    [-0.17, 0.50, 0.65, 0.32, 1.00, -0.69],
    [0.13, -0.35, -0.55, -0.08, -0.69, 1.00]
])

# Heatmap berdampingan dengan wspace lebih kecil untuk memperdekat jarak antar plot
fig = plt.figure(figsize=(14, 6.5))
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.18)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
cbar_ax = fig.add_subplot(gs[0, 2])

# Plot ECG
sns.heatmap(corr_ecg, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1.0, vmax=1.0,
            xticklabels=features, yticklabels=features, ax=ax1, cbar=False,
            annot_kws={"size": 11, "weight": "bold"})
ax1.set_title("4.20 (a) Matriks Korelasi Fitur HRV — Sensor ECG", fontweight="bold", pad=15)

# Plot PPG
sns.heatmap(corr_ppg, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1.0, vmax=1.0,
            xticklabels=features, yticklabels=features, ax=ax2, cbar_ax=cbar_ax,
            annot_kws={"size": 11, "weight": "bold"})
ax2.set_title("4.20 (b) Matriks Korelasi Fitur HRV — Sensor PPG", fontweight="bold", pad=15)

# Konfigurasi label colorbar
cbar_ax.set_ylabel("Koefisien Korelasi Pearson (r)", fontweight="bold", labelpad=12, fontsize=12)

# Bold untuk ticks koordinat pada kedua axis
for ax in [ax1, ax2]:
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
        
for tick in cbar_ax.get_yticklabels():
    tick.set_fontweight("bold")

plt.tight_layout()

# ─── Simpan ───────────────────────────────────────────────────────────────────
FILE_NAME  = "Gambar 4.20 Matriks Korelasi Pearson Antar-Fitur ECG dan PPG.png"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SCRIPT_DIR, "result")
os.makedirs(RESULT_DIR, exist_ok=True)

for dest in [os.path.join(SCRIPT_DIR, FILE_NAME), os.path.join(RESULT_DIR, FILE_NAME)]:
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[OK] Tersimpan: {dest}")
print("[OK] Selesai me-render Gambar 4.20")
