import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

matplotlib.use("Agg")
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['axes.labelsize'] = 13
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['legend.fontsize'] = 12

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ─── Konfigurasi sinyal PPG ────────────────────────────────────────────────────
FS = 256
RAW_PPG_FILE = os.path.join(
    BASE_DIR, "data", "00_raw", "CLAS", "Participants",
    "Part1", "by_block", "1_gsr_ppg_.csv"
)

from scipy.signal import detrend

def apply_moving_average_filter(signal_array, window_size=5):
    filter_weights = np.ones(window_size) / window_size
    return np.convolve(signal_array, filter_weights, mode="same")

def pipeline_filter_lokal(sinyal_raw):
    sinyal_detrend = detrend(sinyal_raw)
    sinyal_smooth  = apply_moving_average_filter(sinyal_detrend, window_size=5)
    mu  = np.mean(sinyal_smooth)
    std = np.std(sinyal_smooth)
    return (sinyal_smooth - mu) / std if std != 0 else sinyal_smooth

# ─── Baca & proses sinyal PPG ──────────────────────────────────────────────────
df = pd.read_csv(RAW_PPG_FILE)
df.columns = df.columns.str.strip()
ppg_raw_penuh = df["ppg"].values.astype(float)

N_TAMPIL = FS * 10
OFFSET   = FS * 5
ppg_raw_seg  = ppg_raw_penuh[OFFSET : OFFSET + N_TAMPIL]
ppg_filtered = pipeline_filter_lokal(ppg_raw_seg)

# ─── Deteksi peaks (neurokit2 → scipy fallback) ────────────────────────────────
ppg_peaks_idx = np.array([], dtype=int)
try:
    import neurokit2 as nk
    ppg_clean_nk  = nk.ppg_clean(ppg_raw_seg, sampling_rate=FS)
    info_ppg      = nk.ppg_findpeaks(ppg_clean_nk, sampling_rate=FS)
    ppg_peaks_idx = np.array(info_ppg.get("PPG_Peaks", []), dtype=int)
    ppg_peaks_idx = ppg_peaks_idx[ppg_peaks_idx < len(ppg_filtered)]
except ImportError:
    from scipy.signal import find_peaks
    ppg_peaks_idx, _ = find_peaks(ppg_filtered, distance=int(FS * 0.35), height=0.2)

n_peaks = len(ppg_peaks_idx)

t = np.arange(N_TAMPIL) / FS
ppg_raw_norm = (ppg_raw_seg - np.mean(ppg_raw_seg)) / np.std(ppg_raw_seg)

# ─── Plotting ──────────────────────────────────────────────────────────────────
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_facecolor("white")
ax_left.set_facecolor("#F5F5F5")
ax_right.set_facecolor("#F5F5F5")

# ── Panel Kiri: Sinyal PPG Mentah ──────────────────────────────────────────────
ax_left.plot(t, ppg_raw_norm, color="#C0392B", lw=1.3, label="PPG Mentah (Raw)")

# Anotasi artefak/spike jika ada nilai ekstrem
spike_idx = int(np.argmax(np.abs(ppg_raw_norm)))
spike_y   = ppg_raw_norm[spike_idx]
spike_x   = t[spike_idx]
if abs(spike_y) > 2.0:
    ax_left.annotate(
        "Artefak / Spike",
        xy=(spike_x, spike_y),
        xytext=(spike_x + 1.5, spike_y * 0.6),
        fontweight="bold", fontsize=12,
        color="#C0392B",
        arrowprops=dict(
            arrowstyle="->",
            color="#C0392B",
            lw=1.5,
            connectionstyle="arc3,rad=-0.2"
        )
    )

ax_left.set_title("4.1 (a)  Sebelum Pemrosesan — Raw Signal", pad=10, fontweight="bold", loc="left")
ax_left.set_xlabel("Waktu (detik)", labelpad=6)
ax_left.set_ylabel("Amplitudo (a.u.)", labelpad=6)
ax_left.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
ax_left.legend(loc="upper right")
ax_left.spines["top"].set_visible(False)
ax_left.spines["right"].set_visible(False)

# ── Panel Kanan: Sinyal PPG Terfilter ──────────────────────────────────────────
ax_right.plot(t, ppg_filtered, color="#154360", lw=1.3,
              label="PPG Bersih (Filtered + Z-Score)")
if n_peaks > 0:
    ax_right.scatter(
        t[ppg_peaks_idx], ppg_filtered[ppg_peaks_idx],
        color="#6C3483", s=90, zorder=5, marker="^",
        label=f"PPG Peaks  (n={n_peaks}, nk.ppg_findpeaks)"
    )
    for idx in ppg_peaks_idx:
        ax_right.axvline(x=t[idx], color="#6C3483", alpha=0.15, linewidth=0.8, linestyle="--")

ax_right.set_title("4.1 (b)  Setelah Filtering & Normalisasi", pad=10, fontweight="bold", loc="left")
ax_right.set_xlabel("Waktu (detik)", labelpad=6)
ax_right.set_ylabel("Amplitudo (Z-Score)", labelpad=14)
ax_right.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
ax_right.legend(loc="upper right")
ax_right.spines["top"].set_visible(False)
ax_right.spines["right"].set_visible(False)

# ── Garis pemisah vertikal di tengah ──────────────────────────────────────────
fig.add_artist(mlines.Line2D(
    [0.505, 0.505], [0.08, 0.94],
    transform=fig.transFigure,
    color="#AAAAAA", linewidth=1.0, linestyle="--"
))

plt.tight_layout()
plt.subplots_adjust(wspace=0.40)

# ─── Simpan ────────────────────────────────────────────────────────────────────
FILE_NAME  = "Gambar 4.1 Visualisasi Pipeline Pemrosesan Sinyal PPG.png"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SCRIPT_DIR, "result")
os.makedirs(RESULT_DIR, exist_ok=True)

for dest in [os.path.join(SCRIPT_DIR, FILE_NAME), os.path.join(RESULT_DIR, FILE_NAME)]:
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[OK] Tersimpan: {dest}")
print(f"[OK] Selesai me-render Gambar 4.1 — {n_peaks} PPG peaks terdeteksi")
