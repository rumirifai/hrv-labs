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
matplotlib.rcParams['ytick.labelsize'] = 11
matplotlib.rcParams['legend.fontsize'] = 13

# Dapatkan BASE_DIR dari letak skrip (mengarah ke root workspace d:\hrv-labs)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

features = [
    "MeanNN\n(ms)",
    "SDNN (ms)",
    "RMSSD (ms)",
    "LF Power\n(nu)",
    "HF Power (nu)",
    "LF/HF Ratio"
]
N = len(features)

# Data persentase kontribusi asli dari penelitian Gambar 4.19
ufr_ecg = [0.4, 30.5, 31.6, 27.7, 3.1, 6.7]
mrmr_ecg = [0.4, 27.2, 29.3, 30.0, 4.3, 8.7]
rf_ecg = [21.0, 20.6, 20.6, 15.6, 12.3, 9.9]

ufr_ppg = [26.3, 1.3, 0.2, 5.0, 50.3, 17.0]
mrmr_ppg = [57.7, 0.6, 0.1, 8.5, 23.7, 9.4]
rf_ppg = [18.7, 22.0, 19.9, 13.7, 14.3, 11.5]

# Penambahan nilai pertama ke ujung radar untuk sirkulasi tertutup
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

val_ufr_ecg = ufr_ecg + ufr_ecg[:1]
val_mrmr_ecg = mrmr_ecg + mrmr_ecg[:1]
val_rf_ecg = rf_ecg + rf_ecg[:1]

val_ufr_ppg = ufr_ppg + ufr_ppg[:1]
val_mrmr_ppg = mrmr_ppg + mrmr_ppg[:1]
val_rf_ppg = rf_ppg + rf_ppg[:1]

# Perbesar figure untuk menjaga jarak antar lingkaran agar tidak saling tabrak
fig = plt.figure(figsize=(15, 7.5))
fig.patch.set_facecolor("white")

# Konfigurasi plotting Radar Chart
def plot_radar(ax, val_ufr, val_mrmr, val_rf, title, rlim_val=45, rticks_vals=[10, 20, 30, 40]):
    ax.set_facecolor("#F8F9F9")
    
    # Plot data
    ax.plot(angles, val_ufr, linewidth=2.0, linestyle="solid", label="UFR", color="#2E86C1")
    ax.fill(angles, val_ufr, color="#2E86C1", alpha=0.1)
    
    ax.plot(angles, val_mrmr, linewidth=2.0, linestyle="solid", label="mRMR", color="#E74C3C")
    ax.fill(angles, val_mrmr, color="#E74C3C", alpha=0.1)
    
    ax.plot(angles, val_rf, linewidth=2.0, linestyle="solid", label="RF Importance", color="#27AE60")
    ax.fill(angles, val_rf, color="#27AE60", alpha=0.1)
    
    # Judul subplot
    ax.set_title(title, fontweight="bold", pad=25)
    
    # Set tick labels dan berikan padding agar tidak menabrak lingkaran
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontweight="bold")
    ax.tick_params(axis='x', pad=15)
    
    # Batas radial dan label grid
    ax.set_rlim(0, rlim_val)
    ax.set_rticks(rticks_vals)
    ax.set_yticklabels([f"{tick}%" for tick in rticks_vals], fontsize=10, fontweight="bold", color="grey")

# Panel A: ECG
ax1 = fig.add_subplot(121, polar=True)
plot_radar(ax1, val_ufr_ecg, val_mrmr_ecg, val_rf_ecg, "4.21 (a) Sensor ECG", rlim_val=45, rticks_vals=[10, 20, 30, 40])

# Panel B: PPG
ax2 = fig.add_subplot(122, polar=True)
plot_radar(ax2, val_ufr_ppg, val_mrmr_ppg, val_rf_ppg, "4.21 (b) Sensor PPG", rlim_val=62, rticks_vals=[10, 20, 30, 40, 50])

# Gunakan wspace yang cukup lebar agar lingkaran tidak bertabrakan
plt.subplots_adjust(wspace=0.35)

# Legend global di bagian bawah tengah
legend_patches = [
    mpatches.Patch(color="#2E86C1", label="UFR"),
    mpatches.Patch(color="#E74C3C", label="mRMR"),
    mpatches.Patch(color="#27AE60", label="RF Importance"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=13, framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.05, 1, 1.0])

# ─── Simpan ───────────────────────────────────────────────────────────────────
FILE_NAME  = "Gambar 4.21 Radar Chart Kontribusi Fitur HRV Secara Agregat.png"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SCRIPT_DIR, "result")
os.makedirs(RESULT_DIR, exist_ok=True)

for dest in [os.path.join(SCRIPT_DIR, FILE_NAME), os.path.join(RESULT_DIR, FILE_NAME)]:
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[OK] Tersimpan: {dest}")
print("[OK] Selesai me-render Gambar 4.21")
