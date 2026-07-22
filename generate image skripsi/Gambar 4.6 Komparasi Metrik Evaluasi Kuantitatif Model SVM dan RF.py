import os
import sys
import numpy as np
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

# Data dari Gambar 4.6 orisinil
metrics_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
ecg_svm = [58.3, 56.1, 64.0, 59.5]
ecg_rf  = [58.6, 57.5, 58.0, 57.4]

ppg_svm = [59.3, 59.8, 61.0, 59.5]
ppg_rf  = [60.4, 59.8, 62.7, 60.8]

x = np.arange(len(metrics_labels))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_facecolor("white")
ax1.set_facecolor("#F8F9F9")
ax2.set_facecolor("#F8F9F9")

# ── Panel Kiri: ECG Tabular ───────────────────────────────────────────────────
rects1_a = ax1.bar(x - width/2, ecg_svm, width, label="SVM Baseline", color="#2980B9")
rects1_b = ax1.bar(x + width/2, ecg_rf, width, label="Random Forest Baseline", color="#27AE60")

ax1.set_ylabel("Nilai (%)", labelpad=8)
ax1.set_title("4.6 (a)  Perbandingan Metrik Klasik - ECG Tabular", pad=12, fontweight="bold", loc="left")
ax1.set_xticks(x)
ax1.set_xticklabels(metrics_labels)
ax1.set_ylim(0, 100)
ax1.grid(True, axis="y", linestyle=":", alpha=0.6)
ax1.legend(loc="upper center", ncol=2, framealpha=0.9)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# ── Panel Kanan: PPG Tabular ──────────────────────────────────────────────────
rects2_a = ax2.bar(x - width/2, ppg_svm, width, label="SVM Baseline", color="#2980B9")
rects2_b = ax2.bar(x + width/2, ppg_rf, width, label="Random Forest Baseline", color="#27AE60")

ax2.set_ylabel("Nilai (%)", labelpad=8)
ax2.set_title("4.6 (b)  Perbandingan Metrik Klasik - PPG Tabular", pad=12, fontweight="bold", loc="left")
ax2.set_xticks(x)
ax2.set_xticklabels(metrics_labels)
ax2.set_ylim(0, 100)
ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
ax2.legend(loc="upper center", ncol=2, framealpha=0.9)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Fungsi untuk menambahkan nilai di atas bar
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # offset vertikal 3 poin
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

autolabel(rects1_a, ax1)
autolabel(rects1_b, ax1)
autolabel(rects2_a, ax2)
autolabel(rects2_b, ax2)

plt.tight_layout()

# ─── Simpan ───────────────────────────────────────────────────────────────────
FILE_NAME  = "Gambar 4.6 Komparasi Metrik Evaluasi Kuantitatif Model SVM dan RF.png"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SCRIPT_DIR, "result")
os.makedirs(RESULT_DIR, exist_ok=True)

for dest in [os.path.join(SCRIPT_DIR, FILE_NAME), os.path.join(RESULT_DIR, FILE_NAME)]:
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[OK] Tersimpan: {dest}")
print("[OK] Selesai me-render Gambar 4.6")
