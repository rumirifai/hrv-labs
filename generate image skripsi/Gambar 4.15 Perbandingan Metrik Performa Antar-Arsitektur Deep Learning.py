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

models_labels = [
    "CNN-LSTM Baseline\n(Unweighted, V2)",
    "CNN-LSTM Tuned\n(Unweighted, V2)",
    "CNN-BiLSTM V2\n(Weighted, V2)",
    "CNN-BiLSTM V3\n(Weighted, V3)"
]
accuracy = [54.59, 65.07, 52.25, 64.87]
precision = [53.82, 62.63, 52.24, 67.53]
recall = [62.33, 82.13, 100.00, 67.05]
f1_score = [57.76, 71.07, 68.63, 67.29]

x = np.arange(len(models_labels))
width = 0.18

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("#F8F9F9")

rects1 = ax.bar(x - 1.5*width, accuracy, width, label="Accuracy", color="#2E86C1")
rects2 = ax.bar(x - 0.5*width, precision, width, label="Precision", color="#27AE60")
rects3 = ax.bar(x + 0.5*width, recall, width, label="Recall", color="#F1C40F")
rects4 = ax.bar(x + 1.5*width, f1_score, width, label="F1-Score", color="#E67E22")

ax.set_ylabel("Nilai Persentase (%)", labelpad=8)
ax.set_title("Perbandingan Metrik Performa Antar-Arsitektur Deep Learning", pad=15, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models_labels)
ax.set_ylim(0, 115)
ax.grid(True, axis="y", linestyle=":", alpha=0.6)
ax.legend(loc="upper right", ncol=4, framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

plt.tight_layout()

# ─── Simpan ───────────────────────────────────────────────────────────────────
FILE_NAME  = "Gambar 4.15 Perbandingan Metrik Performa Antar-Arsitektur Deep Learning.png"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SCRIPT_DIR, "result")
os.makedirs(RESULT_DIR, exist_ok=True)

for dest in [os.path.join(SCRIPT_DIR, FILE_NAME), os.path.join(RESULT_DIR, FILE_NAME)]:
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[OK] Tersimpan: {dest}")
print("[OK] Selesai me-render Gambar 4.15")
