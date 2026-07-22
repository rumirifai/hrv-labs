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

studies = [
    ("Singh & Bura (2025)\nWhite-Box Deep Learning\n(EDA+PPG, multimodal)", 93.5, "#7D3C98"),
    ("Qin & Zhong (2025)\nCNN+LSTM Wearable\n(Multimodal, N=40)", 88.4, "#7D3C98"),
    ("Literature Benchmark\nSVM/RF HRV Tabular\n(54%-84% range)", 70.0, "#95A5A6"),
    ("CNN-LSTM Tuned\n[Hasil Eksperimen]\nECG Waveform, V2", 65.07, "#2E86C1"),
    ("CNN-LSTM Baseline PPG\n[Hasil Eksperimen]\nPPG Tabular", 65.17, "#2E86C1"),
    ("CNN-BiLSTM V3\n[Hasil Eksperimen]\nECG Waveform, V3 + Cosine LR", 64.87, "#2E86C1"),
    ("LSTM+Attention ECG\n[Hasil Eksperimen]\nECG Tabular", 63.43, "#27AE60"),
    ("SVM Baseline\n[Hasil Eksperimen]\nECG Tabular", 63.00, "#27AE60"),
]

labels = [s[0] for s in studies]
accs = [s[1] for s in studies]
colors = [s[2] for s in studies]

fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor("white")
ax.set_facecolor("#F8F9F9")

y = np.arange(len(labels))
bars = ax.barh(y, accs, 0.55, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)

# Label akurasi di ujung bar (dibuat bold)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{acc:.1f}%", va="center", ha="left", fontsize=10, fontweight="bold")

# Hanya menyisakan garis target benchmark 65% (garis merah 60% dihapus)
ax.axvline(x=65, color="#F39C12", linestyle=":", alpha=0.7, lw=1.2, label="Target Benchmark (>= 65%)")
ax.axhline(y=2.5, color="#AAAAAA", linestyle="-", alpha=0.5, lw=1.0)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=11, fontweight="bold")
ax.set_xlabel("Akurasi (%)", labelpad=8, fontweight="bold")
ax.set_xlim(40, 105)
ax.grid(True, axis="x", linestyle=":", alpha=0.6)

# Judul diletakkan di axes title
ax.set_title("Komparasi Akurasi Klasifikasi Beban Kognitif: Hasil Eksperimen vs. Studi Terdahulu", pad=15, fontweight="bold")

# Legend ditaruh di kanan atas (upper right)
legend_patches = [
    mpatches.Patch(color="#7D3C98", label="Studi Terdahulu (Deep Learning Multimodal)"),
    mpatches.Patch(color="#95A5A6", label="Benchmark Literatur (SVM/RF HRV)"),
    mpatches.Patch(color="#2E86C1", label="Model Waveform Terbaik [Hasil Eksperimen]"),
    mpatches.Patch(color="#27AE60", label="Model Tabular Terbaik [Hasil Eksperimen]"),
]
ax.legend(handles=legend_patches, loc="upper right", fontsize=11, framealpha=0.9)

# Bold untuk koordinat (xticks dan yticks)
for label in ax.get_xticklabels():
    label.set_fontweight("bold")
    label.set_fontsize(11)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

# ─── Simpan ───────────────────────────────────────────────────────────────────
FILE_NAME  = "Gambar 4.18 Komparasi Akurasi Klasifikasi Beban Kognitif.png"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SCRIPT_DIR, "result")
os.makedirs(RESULT_DIR, exist_ok=True)

for dest in [os.path.join(SCRIPT_DIR, FILE_NAME), os.path.join(RESULT_DIR, FILE_NAME)]:
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[OK] Tersimpan: {dest}")
print("[OK] Selesai me-render Gambar 4.18")
