import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['axes.titlesize'] = 16
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['legend.fontsize'] = 13

# Dapatkan BASE_DIR dari letak skrip (mengarah ke root workspace d:\hrv-labs)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Fitur penting & data persentase asli dari gambar skripsi
x_labels = ["MeanNN\n(ms)", "SDNN\n(ms)", "RMSSD\n(ms)", "LF Power\n(nu)", "HF Power\n(nu)", "LF/HF\nRatio"]

ufr_ecg = [0.4, 30.5, 31.6, 27.7, 3.1, 6.7]
ufr_ppg = [26.3, 1.3, 0.2, 5.0, 50.3, 17.0]

mrmr_ecg = [0.4, 27.2, 29.3, 30.0, 4.3, 8.7]
mrmr_ppg = [57.7, 0.6, 0.1, 8.5, 23.7, 9.4]

rf_ecg = [21.0, 20.6, 20.6, 15.6, 12.3, 9.9]
rf_ppg = [18.7, 22.0, 19.9, 13.7, 14.3, 11.5]

# Warna bar berdasarkan Ranah Waktu (3 pertama: biru) dan Ranah Frekuensi (3 terakhir: hijau)
bar_colors = ["#2E86C1", "#2E86C1", "#2E86C1", "#27AE60", "#27AE60", "#27AE60"]

# Setup Grid 3x2 - adjusted figsize to make plots look bigger relative to text
fig, axes = plt.subplots(3, 2, figsize=(12, 15))
fig.patch.set_facecolor("white")

for row in range(3):
    for col in range(2):
        axes[row, col].set_facecolor("#F8F9F9")

x_pos = np.arange(len(x_labels))

# Helper function untuk plotting dan labeling
def plot_subplot(ax, data, title, is_left=False, ylim_val=40):
    bars = ax.bar(x_pos, data, color=bar_colors, edgecolor="none", width=0.55, alpha=0.9)
    ax.set_title(title, pad=12, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontweight="bold")
    ax.set_ylim(0, ylim_val)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    
    if is_left:
        ax.set_ylabel("Skor Kontribusi (%)", fontweight="bold", labelpad=8)
        
    # Tambahkan anotasi persentase di atas bar (dibuat lebih besar)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (ylim_val * 0.015),
                f"{height:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    # Bold dan rapikan ticks koordinat
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

# Baris 0: UFR
plot_subplot(axes[0, 0], ufr_ecg, "4.19 (a) ECG — UFR (ANOVA F-Value)", is_left=True, ylim_val=40)
plot_subplot(axes[0, 1], ufr_ppg, "4.19 (b) PPG — UFR (ANOVA F-Value)", is_left=False, ylim_val=68)

# Baris 1: mRMR
plot_subplot(axes[1, 0], mrmr_ecg, "4.19 (c) ECG — mRMR", is_left=True, ylim_val=40)
plot_subplot(axes[1, 1], mrmr_ppg, "4.19 (d) PPG — mRMR", is_left=False, ylim_val=75)

# Baris 2: Random Forest Gini Importance (Gelar 1 baris saja)
plot_subplot(axes[2, 0], rf_ecg, "4.19 (e) ECG — Random Forest Gini Importance", is_left=True, ylim_val=28)
plot_subplot(axes[2, 1], rf_ppg, "4.19 (f) PPG — Random Forest Gini Importance", is_left=False, ylim_val=28)

# Hapus spines atas dan kanan untuk estetika premium
for ax in axes.flat:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Tambahkan legend global di bagian bawah
legend_patches = [
    mpatches.Patch(color="#2E86C1", label="Ranah Waktu (Time Domain)"),
    mpatches.Patch(color="#27AE60", label="Ranah Frekuensi (Frequency Domain)"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=2, fontsize=13, framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

# Berikan ruang untuk legend global di bagian bawah
plt.tight_layout(rect=[0, 0.03, 1, 1.0])

# ─── Simpan ───────────────────────────────────────────────────────────────────
FILE_NAME  = "Gambar 4.19 Komparasi Skor Kontribusi Fitur HRV pada ECG dan PPG.png"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SCRIPT_DIR, "result")
os.makedirs(RESULT_DIR, exist_ok=True)

for dest in [os.path.join(SCRIPT_DIR, FILE_NAME), os.path.join(RESULT_DIR, FILE_NAME)]:
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[OK] Tersimpan: {dest}")
print("[OK] Selesai me-render Gambar 4.19")
