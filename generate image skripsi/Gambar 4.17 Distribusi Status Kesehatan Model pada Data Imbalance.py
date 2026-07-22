import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
matplotlib.rcParams['font.size'] = 13
matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['axes.labelsize'] = 13
matplotlib.rcParams['xtick.labelsize'] = 11
matplotlib.rcParams['ytick.labelsize'] = 11
matplotlib.rcParams['legend.fontsize'] = 11

# Dapatkan BASE_DIR dari letak skrip (mengarah ke root workspace d:\hrv-labs)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Bar plot breakdown data
cat_labels = ["ECG Tabular\n(8 model)", "PPG Tabular\n(5 model)", "ECG Waveform\n(3 model)"]
sehat_vals = [6, 4, 2]
cukup_vals = [0, 0, 0]
anomali_vals = [2, 1, 1]

x = np.arange(len(cat_labels))
w = 0.25

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("#F8F9F9")

# Bar plot
ax.bar(x - w, sehat_vals, w, label="SEHAT", color="#2E86C1")
ax.bar(x, cukup_vals, w, label="Cukup", color="#F39C12")
ax.bar(x + w, anomali_vals, w, label="ANOMALI", color="#C0392B")

ax.set_xticks(x)
ax.set_xticklabels(cat_labels, fontweight="bold")
ax.set_ylabel("Jumlah Model", labelpad=8)
ax.set_title("Distribusi Kesehatan Model per Kategori Sensor", pad=15, fontweight="bold")
ax.grid(True, axis="y", linestyle=":", alpha=0.6)
ax.legend(loc="upper right", framealpha=0.9)
ax.set_ylim(0, 8)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

# ─── Simpan ───────────────────────────────────────────────────────────────────
FILE_NAME  = "Gambar 4.17 Distribusi Status Kesehatan Model pada Data Imbalance.png"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SCRIPT_DIR, "result")
os.makedirs(RESULT_DIR, exist_ok=True)

for dest in [os.path.join(SCRIPT_DIR, FILE_NAME), os.path.join(RESULT_DIR, FILE_NAME)]:
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[OK] Tersimpan: {dest}")
print("[OK] Selesai me-render Gambar 4.17")
