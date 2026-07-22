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
matplotlib.rcParams['axes.titlesize'] = 15
matplotlib.rcParams['axes.labelsize'] = 13
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['legend.fontsize'] = 11

# Dapatkan BASE_DIR dari letak skrip (mengarah ke root workspace d:\hrv-labs)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

models_data = [
    ("CNN-LSTM Baseline",      "PPG Tabular",    65.17, 64.16, "SEHAT"),
    ("CNN-LSTM Tuned",         "ECG Waveform",   65.07, 71.07, "SEHAT"),
    ("CNN-BiLSTM V3",          "ECG Waveform",   64.87, 67.29, "SEHAT"),
    ("CNN-LSTM Baseline",      "ECG Tabular",    64.00, 65.38, "SEHAT"),
    ("LSTM + Attention",       "ECG Tabular",    63.43, 65.96, "SEHAT"),
    ("SVM Baseline",           "ECG Tabular",    63.00, 47.00, "SEHAT"),
    ("RF Baseline",            "ECG Tabular",    63.00, 47.00, "SEHAT"),
    ("RF Baseline",            "PPG Tabular",    62.00, 46.00, "SEHAT"),
    ("SVM Baseline",           "PPG Tabular",    61.00, 45.00, "SEHAT"),
    ("CNN-LSTM + mRMR",        "ECG Tabular",    60.29, 55.02, "SEHAT"),
    ("LSTM + Attention",       "PPG Tabular",    60.11, 61.41, "SEHAT"),
    ("CNN-LSTM + UFR",         "ECG Tabular",    60.00, 54.84, "SEHAT"),
    ("Fusi Multimodal",        "ECG+PPG",        55.43, 61.00, "CUKUP"),
    ("CNN-BiLSTM V2",          "ECG Waveform",   52.25, 68.63, "ANOMALI"),
    ("CNN-LSTM + UFR",         "PPG Tabular",    50.56, 66.15, "ANOMALI"),
    ("CNN-LSTM + mRMR",        "PPG Tabular",    50.00, 66.03, "ANOMALI"),
]

labels = [f"{m[0]}\n({m[1]})" for m in models_data]
accuracy = [m[2] for m in models_data]
f1scores = [m[3] for m in models_data]
statuses = [m[4] for m in models_data]

color_map = {"SEHAT": "#2E86C1", "CUKUP": "#F39C12", "ANOMALI": "#C0392B"}
acc_colors = [color_map[s] for s in statuses]

y = np.arange(len(labels))
height = 0.35

fig, ax = plt.subplots(figsize=(13, 10))
fig.patch.set_facecolor("white")
ax.set_facecolor("#F8F9F9")

bars1 = ax.barh(y + height/2, accuracy, height, label="Accuracy (%)", color=acc_colors, alpha=0.85)
bars2 = ax.barh(y - height/2, f1scores, height, label="F1-Score (%)", color=acc_colors, alpha=0.4, hatch="//")

# Tambahkan label persentase (dibuat bold dan ukuran 10 agar jelas)
for bar in bars1:
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{bar.get_width():.1f}%", va="center", ha="left", fontsize=10, fontweight="bold")
for bar in bars2:
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{bar.get_width():.1f}%", va="center", ha="left", fontsize=10, fontweight="bold", alpha=0.7)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=11, fontweight="bold")
ax.set_xlabel("Nilai Persentase (%)", labelpad=8, fontweight="bold")
ax.set_xlim(30, 88)
ax.grid(True, axis="x", linestyle=":", alpha=0.6)

# Judul diletakkan di axes title
ax.set_title("Papan Peringkat Komprehensif Performa Seluruh Model Eksperimen", pad=20, fontweight="bold")

# Legend di bagian atas (upper right) dan menghapus ikon/emoji
legend_patches = [
    mpatches.Patch(color="#2E86C1", label="SEHAT (Stabil, Acc >= 60%)"),
    mpatches.Patch(color="#F39C12", label="Cukup / Valid"),
    mpatches.Patch(color="#C0392B", label="ANOMALI (Bias Recall)"),
    mpatches.Patch(facecolor="gray", alpha=0.4, hatch="//", label="F1-Score"),
]
ax.legend(handles=legend_patches, loc="upper right", fontsize=11, framealpha=0.9)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

# ─── Simpan ───────────────────────────────────────────────────────────────────
FILE_NAME  = "Gambar 4.16 Model Leaderboard Performa Seluruh Model Eksperimen.png"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SCRIPT_DIR, "result")
os.makedirs(RESULT_DIR, exist_ok=True)

for dest in [os.path.join(SCRIPT_DIR, FILE_NAME), os.path.join(RESULT_DIR, FILE_NAME)]:
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[OK] Tersimpan: {dest}")
print("[OK] Selesai me-render Gambar 4.16")
