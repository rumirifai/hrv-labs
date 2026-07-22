import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

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

# ─── Konfigurasi sinyal ECG ────────────────────────────────────────────────────
FS = 256
RAW_ECG_FILE = os.path.join(
    BASE_DIR, "data", "00_raw", "CLAS", "Participants",
    "Part1", "by_block", "1_ecg_.csv"
)

from scipy.signal import detrend

# Filter sesuai pipeline penelitian: Detrend → Moving Average (k=5) → Z-Score
def apply_moving_average_filter(signal_array, window_size=5):
    filter_weights = np.ones(window_size) / window_size
    return np.convolve(signal_array, filter_weights, mode="same")

def pipeline_filter_lokal(sinyal_raw):
    sinyal_detrend = detrend(sinyal_raw)
    sinyal_smooth  = apply_moving_average_filter(sinyal_detrend, window_size=5)
    mu  = np.mean(sinyal_smooth)
    std = np.std(sinyal_smooth)
    return (sinyal_smooth - mu) / std if std != 0 else sinyal_smooth

# ─── Baca & proses sinyal ECG (offset 5s, sama seperti original) ──────────────
df = pd.read_csv(RAW_ECG_FILE)
df.columns = df.columns.str.strip()
ecg_raw_penuh = df["ecg2"].values.astype(float)

N_TAMPIL = FS * 10
OFFSET   = FS * 5
ecg_raw_seg  = ecg_raw_penuh[OFFSET : OFFSET + N_TAMPIL]
ecg_filtered = pipeline_filter_lokal(ecg_raw_seg)

# ─── Deteksi R-peaks via neurokit2 / scipy ────────────────────────────────────
ecg_rpeaks_idx = np.array([], dtype=int)
try:
    import neurokit2 as nk
    ecg_clean_nk   = nk.ecg_clean(ecg_raw_seg, sampling_rate=FS)
    _, info_ecg    = nk.ecg_peaks(ecg_clean_nk, sampling_rate=FS)
    ecg_rpeaks_idx = np.array(info_ecg.get("ECG_R_Peaks", []), dtype=int)
    ecg_rpeaks_idx = ecg_rpeaks_idx[ecg_rpeaks_idx < N_TAMPIL]
except ImportError:
    from scipy.signal import find_peaks
    ecg_rpeaks_idx, _ = find_peaks(ecg_filtered, distance=int(FS * 0.35), height=0.5)

n_peaks = len(ecg_rpeaks_idx)

t = np.arange(N_TAMPIL) / FS
ecg_raw_norm = (ecg_raw_seg - np.mean(ecg_raw_seg)) / np.std(ecg_raw_seg)

# ─── Tentukan y-axis range yang sama untuk kedua panel ───────────────────────
# Gunakan range sinyal yang terdeteksi, di-clip ke ±3 agar simetris & rapi
y_min_raw = max(np.min(ecg_raw_norm), -3.0)
y_max_raw = min(np.max(ecg_raw_norm),  3.0)
y_min_fil = max(np.min(ecg_filtered),  -3.0)
y_max_fil = min(np.max(ecg_filtered),   3.0)
y_shared_min = min(y_min_raw, y_min_fil) - 0.1
y_shared_max = max(y_max_raw, y_max_fil) + 0.1
# Bulatkan agar tick rapi
y_shared_min = np.floor(y_shared_min * 2) / 2
y_shared_max = np.ceil(y_shared_max  * 2) / 2

# ─── Plotting ─────────────────────────────────────────────────────────────────
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_facecolor("white")
ax_left.set_facecolor("#F5F5F5")
ax_right.set_facecolor("#F5F5F5")

# ── Panel Kiri: Sinyal ECG Mentah ─────────────────────────────────────────────
ax_left.plot(t, ecg_raw_norm, color="#6E2C00", lw=1.3, label="ECG Mentah (Raw)")

# Anotasi Baseline Wander (menunjuk ke area transisi drift di awal sinyal)
ann_idx = int(1.5 * FS)
ann_y   = float(np.clip(ecg_raw_norm[ann_idx], y_shared_min + 0.2, y_shared_max - 0.8))
ann_x   = t[ann_idx]
ax_left.annotate(
    "Baseline Wander",
    xy=(ann_x, ann_y),
    xytext=(ann_x + 2.0, ann_y + 1.0),
    fontweight="bold", fontsize=12,
    color="#6E2C00",
    arrowprops=dict(
        arrowstyle="->",
        color="#6E2C00",
        lw=1.5,
        connectionstyle="arc3,rad=0.25"
    )
)

ax_left.set_title("4.2 (a)  Sebelum Pemrosesan — Raw Signal", pad=10, fontweight="bold", loc="left")
ax_left.set_xlabel("Waktu (detik)", labelpad=6)
ax_left.set_ylabel("Amplitudo (a.u.)", labelpad=6)
ax_left.set_ylim(y_shared_min, y_shared_max)
ax_left.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
ax_left.legend(loc="upper right")
ax_left.spines["top"].set_visible(False)
ax_left.spines["right"].set_visible(False)

# ── Panel Kanan: Sinyal ECG Terfilter + Deteksi R-Peak ────────────────────────
ax_right.plot(t, ecg_filtered, color="#145A32", lw=1.3,
              label="ECG Bersih (Filtered + Z-Score)")
if n_peaks > 0:
    ax_right.scatter(
        t[ecg_rpeaks_idx], ecg_filtered[ecg_rpeaks_idx],
        color="#7D6608", s=90, zorder=5, marker="v",
        label=f"ECG R-Peaks  (n={n_peaks}, nk.ecg_peaks)"
    )
    for idx in ecg_rpeaks_idx:
        ax_right.axvline(x=t[idx], color="#7D6608", alpha=0.15, linewidth=0.8, linestyle="--")

ax_right.set_title("4.2 (b)  Setelah Filtering & Deteksi R-Peak", pad=10, fontweight="bold", loc="left")
ax_right.set_xlabel("Waktu (detik)", labelpad=6)
ax_right.set_ylabel("Amplitudo (Z-Score)", labelpad=14)
ax_right.set_ylim(y_shared_min, y_shared_max)
ax_right.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
ax_right.legend(loc="upper right")
ax_right.spines["top"].set_visible(False)
ax_right.spines["right"].set_visible(False)

# ── Garis pemisah vertikal di tengah ─────────────────────────────────────────
fig.add_artist(mlines.Line2D(
    [0.505, 0.505], [0.08, 0.94],
    transform=fig.transFigure,
    color="#AAAAAA", linewidth=1.0, linestyle="--"
))

plt.tight_layout()
plt.subplots_adjust(wspace=0.40)

# ─── Simpan ────────────────────────────────────────────────────────────────────
FILE_NAME  = "Gambar 4.2 Visualisasi Pipeline Pemrosesan Sinyal ECG.png"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SCRIPT_DIR, "result")
os.makedirs(RESULT_DIR, exist_ok=True)

for dest in [os.path.join(SCRIPT_DIR, FILE_NAME), os.path.join(RESULT_DIR, FILE_NAME)]:
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[OK] Tersimpan: {dest}")
print(f"[OK] Selesai me-render Gambar 4.2 — {n_peaks} ECG R-peaks terdeteksi")
