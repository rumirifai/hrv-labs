import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.use("Agg")
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
matplotlib.rcParams['font.size'] = 15
matplotlib.rcParams['axes.titlesize'] = 15
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['legend.fontsize'] = 13

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ─── Konfigurasi ───────────────────────────────────────────────────────────────
FS       = 256
WIN_SEC  = 5
WIN_PTS  = int(FS * WIN_SEC)
GEN_SEC  = 2.5 # dummy or similar
STEP_SEC = 2.5
STEP_PTS = int(FS * STEP_SEC)
RAW_ECG_FILE = os.path.join(
    BASE_DIR, "data", "00_raw", "CLAS", "Participants",
    "Part1", "by_block", "1_ecg_.csv"
)

from scipy.signal import detrend

def preprocess_window(seg):
    """Pipeline sesuai penelitian: Detrend → MA(k=5) → Z-Score."""
    d  = detrend(seg.astype(float).copy())
    sm = np.convolve(d, np.ones(5) / 5, mode="same")
    mu, std = np.mean(sm), np.std(sm)
    return (sm - mu) / std if std > 0 else sm

# ─── Baca data (pipeline asli, tidak diubah) ──────────────────────────────────
df = pd.read_csv(RAW_ECG_FILE)
df.columns = df.columns.str.strip()
ecg_raw  = df["ecg2"].values.astype(float)
ecg_norm = (ecg_raw - np.mean(ecg_raw)) / np.std(ecg_raw)

N_SHOW_PTS = FS * 14
ecg_show   = ecg_norm[:N_SHOW_PTS]
t_show     = np.arange(len(ecg_show)) / FS

WIN_COL = ["#2471A3", "#C0392B", "#1E8449", "#884EA0"]
WIN_LABELS = ["W1", "W2", "W3", "W4"]

# ─── Layout: 1 besar atas + 2×2 kecil bawah ──────────────────────────────────
# GridSpec: baris 0 = panel atas (full width), baris 1-2 = 2×2 panel bawah
fig = plt.figure(figsize=(14, 11))
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(
    3, 4,
    height_ratios=[1.6, 1, 1],
    hspace=0.48, wspace=0.38
)

# ── Panel Atas (full width): Sinyal ECG Kontinu ────────────────────────────────
ax_top = fig.add_subplot(gs[0, :])
ax_top.set_facecolor("#F5F5F5")
ax_top.plot(t_show, ecg_show, color="#6E2C00", lw=1.0, label="Sinyal ECG Mentah (Raw)")

# Overlay jendela W1–W4 di panel atas
for idx in range(4):
    pos = int(idx * STEP_PTS)
    t_s = pos / FS
    t_e = (pos + WIN_PTS) / FS
    color = WIN_COL[idx]
    ax_top.axvspan(t_s, t_e, alpha=0.15, color=color)
    ax_top.axvline(t_s, color=color, ls="--", alpha=0.6, lw=1.2)
    ax_top.axvline(t_e, color=color, ls="--", alpha=0.6, lw=1.2)
    # Label W1–W4 di tengah jendela, dekat atas plot
    y_top = ax_top.get_ylim()[1] if ax_top.get_ylim()[1] != 0 else 1.5
    ax_top.text(
        (t_s + t_e) / 2, 1.4,
        WIN_LABELS[idx],
        color=color, fontweight="bold", ha="center", fontsize=13
    )

# Anotasi overlap
ax_top.annotate(
    "Overlap 50%\n(2.5s)",
    xy=(2.5, 0.5),
    xytext=(3.4, 0.8),
    fontsize=10, color="#555555", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0)
)

ax_top.set_title("4.3 (a)  Sinyal ECG Kontinu (14 detik)", pad=10, fontweight="bold", loc="left")
ax_top.set_xlabel("Waktu (detik)", labelpad=5)
ax_top.set_ylabel("Amplitudo (a.u.)", labelpad=6)
ax_top.legend(loc="upper right")
ax_top.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
ax_top.spines["top"].set_visible(False)
ax_top.spines["right"].set_visible(False)

# ── 4 Panel Kecil Bawah (2×2): per Jendela ────────────────────────────────────
SUB_POSITIONS = [(1, 0), (1, 2), (2, 0), (2, 2)]   # (baris, kolom-mulai)
SUB_LABELS    = ["b", "c", "d", "e"]

for idx in range(4):
    pos   = int(idx * STEP_PTS)
    t_s   = pos / FS
    t_e   = (pos + WIN_PTS) / FS
    color = WIN_COL[idx]
    row, col_start = SUB_POSITIONS[idx]

    # Setiap subplot menempati 2 kolom gridspec (dari col_start s.d. col_start+2)
    ax_sub = fig.add_subplot(gs[row, col_start : col_start + 2])
    ax_sub.set_facecolor("#F5F5F5")

    raw_win  = ecg_raw[pos : pos + WIN_PTS]
    filt_win = preprocess_window(raw_win)
    t_win    = np.arange(len(filt_win)) / FS

    ax_sub.plot(t_win, filt_win, color=color, lw=1.2)
    ax_sub.axhspan(-1, 1, alpha=0.07, color=color)

    # Judul: "4.3 (b) Jendela W1 (0.0s – 5.0s)"
    sub_lbl = SUB_LABELS[idx]
    ax_sub.set_title(
        f"4.3 ({sub_lbl})  Jendela {WIN_LABELS[idx]} ({t_s:.1f}s \u2013 {t_e:.1f}s)",
        pad=8, fontweight="bold", loc="left", color=color, fontsize=13
    )
    ax_sub.set_xlabel("Waktu relatif (s)", labelpad=4)
    ax_sub.set_ylabel("Amplitudo (Z-Score)", labelpad=6)
    ax_sub.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    ax_sub.spines["top"].set_visible(False)
    ax_sub.spines["right"].set_visible(False)

# ─── Simpan ────────────────────────────────────────────────────────────────────
FILE_NAME  = "Gambar 4.3 Segmentasi Sliding Window pada Isyarat Waveform ECG.png"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SCRIPT_DIR, "result")
os.makedirs(RESULT_DIR, exist_ok=True)

for dest in [os.path.join(SCRIPT_DIR, FILE_NAME), os.path.join(RESULT_DIR, FILE_NAME)]:
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[OK] Tersimpan: {dest}")
print("[OK] Selesai me-render Gambar 4.3")
