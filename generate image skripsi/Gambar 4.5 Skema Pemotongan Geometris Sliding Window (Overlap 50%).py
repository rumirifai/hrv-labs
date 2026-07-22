import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# ─── Konfigurasi sliding window ────────────────────────────────────────────────
FS       = 256
WIN_SEC  = 5.0    # durasi jendela (detik)
STEP_SEC = 2.5    # step/shift (detik)
N_WIN    = 5      # jumlah jendela yang ditampilkan

WIN_COL = ["#2471A3", "#C0392B", "#1E8449", "#884EA0", "#117A65"]

# ─── Sinyal dummy (sinusoidal gabungan, mewakili sinyal fisiologis) ─────────────
TOTAL_SEC = WIN_SEC + (N_WIN - 1) * STEP_SEC + 1.0   # sedikit lebih panjang
t_full    = np.linspace(0, TOTAL_SEC, int(TOTAL_SEC * FS))
signal    = (0.6 * np.sin(2 * np.pi * 1.1 * t_full)
           + 0.3 * np.sin(2 * np.pi * 2.7 * t_full)
           + 0.1 * np.random.default_rng(42).standard_normal(len(t_full)))

# ─── Layout: panel atas (sinyal) + panel bawah (diagram Gantt) ────────────────
fig = plt.figure(figsize=(13, 7))
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(2, 1, height_ratios=[1.4, 2.2], hspace=0.40)

# ══════════════════════════════════════════════════════════════════════════════
# Panel Atas  —  4.5 (a): Sinyal Fisiologis Kontinu
# ══════════════════════════════════════════════════════════════════════════════
ax_top = fig.add_subplot(gs[0])
ax_top.set_facecolor("#F5F5F5")
ax_top.plot(t_full, signal, color="#2C3E50", lw=0.9)

# Shading jendela W1–W5 di panel atas
for i in range(N_WIN):
    t_s = i * STEP_SEC
    t_e = t_s + WIN_SEC
    ax_top.axvspan(t_s, t_e, alpha=0.13, color=WIN_COL[i])

ax_top.set_title("4.5 (a)  Sinyal Fisiologis Kontinu (Rekaman Penuh)",
                 pad=8, fontweight="bold", loc="left")
ax_top.set_ylabel("Sinyal\nKontinu", labelpad=6)
ax_top.set_xlim(0, t_full[-1])
ax_top.set_xticks([])
ax_top.spines["top"].set_visible(False)
ax_top.spines["right"].set_visible(False)
ax_top.spines["bottom"].set_visible(False)

# ══════════════════════════════════════════════════════════════════════════════
# Panel Bawah  —  4.5 (b): Diagram Gantt posisi geometris tiap jendela
# ══════════════════════════════════════════════════════════════════════════════
ax_bot = fig.add_subplot(gs[1])
ax_bot.set_facecolor("white")

ROW_H    = 0.7   # tinggi tiap baris Gantt
ROW_GAP  = 0.15  # jarak antar baris
LABEL_X  = -0.7  # posisi x label "Jendela N" (di luar sumbu kiri)

for i in range(N_WIN):
    t_s   = i * STEP_SEC
    t_e   = t_s + WIN_SEC
    color = WIN_COL[i]
    y_bot = (N_WIN - 1 - i) * (ROW_H + ROW_GAP)   # baris teratas = Jendela 1

    # Kotak jendela
    rect = mpatches.FancyBboxPatch(
        (t_s, y_bot), WIN_SEC, ROW_H,
        boxstyle="round,pad=0.03",
        facecolor=color, alpha=0.22,
        edgecolor=color, linewidth=1.5
    )
    ax_bot.add_patch(rect)

    # Teks isi kotak (waktu): geser sedikit ke kanan agar tidak nabrak garis kiri
    ax_bot.text(
        t_s + WIN_SEC / 2, y_bot + ROW_H / 2,
        f"W{i+1} ({t_s:.1f}s - {t_e:.1f}s)",
        ha="center", va="center",
        fontsize=12, fontweight="bold", color=color
    )

    # Label "Jendela N" di luar kotak (kiri)
    ax_bot.text(
        LABEL_X, y_bot + ROW_H / 2,
        f"Jendela {i+1}",
        ha="right", va="center",
        fontsize=13, fontweight="bold", color=color
    )

    # Garis vertikal putus-putus di tepi kiri & kanan kotak
    ax_bot.axvline(t_s, color=color, ls="--", lw=0.9, alpha=0.5,
                   ymin=y_bot / (N_WIN * (ROW_H + ROW_GAP)),
                   ymax=(y_bot + ROW_H) / (N_WIN * (ROW_H + ROW_GAP)))

# ── Shading area overlap (antara W1 dan W2, di baris W2) ─────────────────────
OVLP_X0 = STEP_SEC          # 2.5s
OVLP_X1 = WIN_SEC            # 5.0s
ovlp_y   = (N_WIN - 2) * (ROW_H + ROW_GAP)   # baris Jendela 2
ovlp_patch = mpatches.FancyBboxPatch(
    (OVLP_X0, ovlp_y), OVLP_X1 - OVLP_X0, ROW_H,
    boxstyle="round,pad=0.03",
    facecolor="#884EA0", alpha=0.30,
    edgecolor="#884EA0", linewidth=1.5, linestyle=":"
)
ax_bot.add_patch(ovlp_patch)

# ── Anotasi "Overlap = W – S" di Kanan Atas dengan Panah Melambung ke Atas ───
ax_bot.annotate(
    "Overlap = W - S\n= 2.5s  (50%)",
    xy=((OVLP_X0 + OVLP_X1) / 2, ovlp_y + ROW_H / 2),
    xytext=(9.0, (N_WIN - 1) * (ROW_H + ROW_GAP) + ROW_H / 2),
    fontsize=11, fontweight="bold", color="#884EA0",
    ha="center", va="center",
    bbox=dict(boxstyle="round,pad=0.4", fc="#F9F0FF", ec="#884EA0", lw=1.5),
    arrowprops=dict(
        arrowstyle="->",
        color="#884EA0", lw=1.5,
        connectionstyle="arc3,rad=0.35"
    )
)

# ── Panah anotasi W (Lebar Jendela) ─────────────────────────────────────────
ARROW_Y = -0.25 
ax_bot.annotate("", xy=(WIN_SEC, ARROW_Y), xytext=(0, ARROW_Y),
                arrowprops=dict(arrowstyle="<->", color="#2C3E50", lw=1.8))
ax_bot.text(WIN_SEC / 2, ARROW_Y - 0.12, "W = 5s (1280 sampel)",
            ha="center", va="top", fontsize=11, fontweight="bold", color="#2C3E50")

# ── Panah anotasi S (Step Jendela) ──────────────────────────────────────────
ARROW_Y2 = -0.75 
ax_bot.annotate("", xy=(STEP_SEC, ARROW_Y2), xytext=(0, ARROW_Y2),
                arrowprops=dict(arrowstyle="<->", color="#7F8C8D", lw=1.5))
ax_bot.text(STEP_SEC / 2, ARROW_Y2 - 0.12, "S = 2.5s (640 sampel)",
            ha="center", va="top", fontsize=11, color="#7F8C8D")

# ── Grid vertikal ringan (penanda waktu) ──────────────────────────────────────
for xt in np.arange(0, 17, 2):
    ax_bot.axvline(xt, color="#CCCCCC", lw=0.7, zorder=0)

ax_bot.set_title("4.5 (b)  Posisi Geometris Tiap Jendela Geser terhadap Sinyal Kontinu",
                 pad=8, fontweight="bold", loc="left")
ax_bot.set_xlabel("Waktu (detik)", labelpad=6)
ax_bot.set_xlim(LABEL_X - 0.3, 16.5)

ax_bot.set_ylim(-1.3, N_WIN * (ROW_H + ROW_GAP) + 0.1)
ax_bot.set_yticks([])
ax_bot.spines["top"].set_visible(False)
ax_bot.spines["right"].set_visible(False)
ax_bot.spines["left"].set_visible(False)

plt.tight_layout()

# ─── Simpan ───────────────────────────────────────────────────────────────────
FILE_NAME  = "Gambar 4.5 Skema Pemotongan Geometris Sliding Window (Overlap 50%).png"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SCRIPT_DIR, "result")
os.makedirs(RESULT_DIR, exist_ok=True)

for dest in [os.path.join(SCRIPT_DIR, FILE_NAME), os.path.join(RESULT_DIR, FILE_NAME)]:
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[OK] Tersimpan: {dest}")
print("[OK] Selesai me-render Gambar 4.5")
