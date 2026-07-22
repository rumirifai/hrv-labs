"""
app.py — Flask Backend: Dashboard Visualisasi Penelitian HRV
Sistem Deteksi Beban Kognitif Mahasiswa berbasis HRV
Muhamad Rumi Rifai | 140810220026 | UNPAD
"""

import os, io, math, warnings, json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, Response

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB max upload

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_rank1_ppg.keras")

# ─── NaN-safe JSON helper ─────────────────────────────────────────────────────
def nan_to_none(obj):
    """Recursively replace float NaN/Inf with None so JSON is always valid."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [nan_to_none(v) for v in obj]
    # numpy scalars
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj

def safe_json(data):
    """Return a JSON Response with NaN values replaced by null."""
    cleaned = nan_to_none(data)
    return Response(json.dumps(cleaned, ensure_ascii=False), mimetype="application/json")

# ─── Load model sekali saat startup ──────────────────────────────────────────
model_ppg   = None
scaler_mean = None
scaler_std  = None

def load_model_and_scaler():
    global model_ppg, scaler_mean, scaler_std
    try:
        from tensorflow.keras.models import load_model
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import KNNImputer

        model_ppg = load_model(MODEL_PATH, compile=False)

        df = pd.read_csv(os.path.join(DATA_DIR, "dataset_ppg.csv"))
        FITUR = ["MeanNN_ms", "SDNN_ms", "RMSSD_ms", "LF", "HF", "LF_HF_Ratio"]
        X = df[FITUR].values
        imputer = KNNImputer(n_neighbors=5)
        X_imp = imputer.fit_transform(X)
        sc = StandardScaler()
        sc.fit(X_imp)
        scaler_mean = sc.mean_
        scaler_std  = sc.scale_
        print(f"[OK] Model dan Scaler berhasil dimuat: {MODEL_PATH}")
    except Exception as e:
        print(f"[WARN] Model tidak dapat dimuat: {e}")

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/dataset/<sensor>")
def api_dataset(sensor):
    """Statistik dataset tabular ECG atau PPG — NaN-safe."""
    fname = f"dataset_{sensor.lower()}.csv"
    path  = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        return safe_json({"error": "Dataset tidak ditemukan"}), 404

    df    = pd.read_csv(path)
    FITUR = ["MeanNN_ms", "SDNN_ms", "RMSSD_ms", "LF", "HF", "LF_HF_Ratio"]

    # Kolom tambahan opsional
    extra_cols = []
    if "Subjek" in df.columns: extra_cols.append("Subjek")
    if "Tugas"  in df.columns: extra_cols.append("Tugas")

    # ── Statistik: hanya dari baris non-NaN per kolom ──────────────────────
    stats = {}
    for f in FITUR:
        if f not in df.columns:
            stats[f] = {"mean": None, "std": None, "min": None, "max": None}
            continue
        col = df[f].dropna()
        if len(col) == 0:
            stats[f] = {"mean": None, "std": None, "min": None, "max": None}
        else:
            stats[f] = {
                "mean": round(float(col.mean()), 3),
                "std":  round(float(col.std()),  3),
                "min":  round(float(col.min()),  3),
                "max":  round(float(col.max()),  3),
            }

    # ── Distribusi label ───────────────────────────────────────────────────
    label_counts = {}
    if "Label" in df.columns:
        label_counts = df["Label"].value_counts().to_dict()

    # ── Sample tabel: ganti NaN → None ────────────────────────────────────
    select_cols = extra_cols + ["Label"] + [f for f in FITUR if f in df.columns]
    sample_df   = df[select_cols].head(20).copy()
    # Replace NaN with None (becomes null in JSON)
    sample_df   = sample_df.where(pd.notnull(sample_df), None)
    sample_rows = sample_df.to_dict(orient="records")

    # Round numeric fields safely
    for row in sample_rows:
        for f in FITUR:
            if f in row and row[f] is not None:
                try:
                    row[f] = round(float(row[f]), 3)
                except (TypeError, ValueError):
                    row[f] = None

    return safe_json({
        "total_rows":   int(len(df)),
        "label_counts": label_counts,
        "stats":        stats,
        "sample":       sample_rows,
    })


@app.route("/api/leaderboard")
def api_leaderboard():
    """Performa semua model dari rekap_model.csv — NaN-safe."""
    path = os.path.join(DATA_DIR, "rekap_model.csv")
    if not os.path.exists(path):
        return safe_json({"error": "Rekap model tidak ditemukan"}), 404

    df = pd.read_csv(path)

    # ── Normalisasi nama kolom ─────────────────────────────────────────────
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl == "model":     col_map[c] = "model"
        elif cl == "sensor":  col_map[c] = "sensor"
        elif cl in ("accuracy", "akurasi"):   col_map[c] = "accuracy"
        elif cl == "precision":               col_map[c] = "precision"
        elif cl in ("recall", "sensitivitas"): col_map[c] = "recall"
        elif cl in ("f1_score", "f1 score", "f1score"): col_map[c] = "f1_score"
        elif cl in ("keterangan", "catatan"): col_map[c] = "keterangan"
    df = df.rename(columns=col_map)

    # Ensure required columns exist
    for req in ["model", "sensor", "accuracy", "precision", "recall", "f1_score"]:
        if req not in df.columns:
            df[req] = 0.0

    # Convert to numeric, fill NaN with 0
    for num_col in ["accuracy", "precision", "recall", "f1_score"]:
        df[num_col] = pd.to_numeric(df[num_col], errors="coerce").fillna(0.0)

    df["accuracy_pct"]  = (df["accuracy"]  * 100).round(2)
    df["f1_score_pct"]  = (df["f1_score"]  * 100).round(2)
    df["precision_pct"] = (df["precision"] * 100).round(2)
    df["recall_pct"]    = (df["recall"]    * 100).round(2)

    # ── Rank & status kesehatan ────────────────────────────────────────────
    df_sorted = df.sort_values("accuracy", ascending=False).reset_index(drop=True)
    df_sorted["rank"] = df_sorted.index + 1

    def get_status(row):
        acc = float(row["accuracy"])
        rec = float(row["recall"])
        pre = float(row["precision"])
        if acc >= 0.60 and abs(rec - pre) < 0.25:
            return "SEHAT"
        elif acc < 0.55 or rec > 0.95:
            return "ANOMALI"
        return "CUKUP"

    df_sorted["status"] = df_sorted.apply(get_status, axis=1)

    # Pilih kolom yang dikirim ke frontend
    out_cols = ["rank", "model", "sensor", "accuracy_pct", "precision_pct",
                "recall_pct", "f1_score_pct", "status"]
    if "keterangan" in df_sorted.columns:
        out_cols.append("keterangan")

    records = df_sorted[out_cols].where(
        pd.notnull(df_sorted[out_cols]), None
    ).to_dict(orient="records")

    return safe_json(records)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Prediksi beban kognitif dari nilai HRV manual."""
    if model_ppg is None:
        return safe_json({"error": "Model belum siap. Restart server dan tunggu beberapa detik."}), 503

    data  = request.get_json()
    FITUR = ["MeanNN_ms", "SDNN_ms", "RMSSD_ms", "LF", "HF", "LF_HF_Ratio"]

    try:
        vals = []
        for f in FITUR:
            v = data.get(f)
            vals.append(float(v) if v not in (None, "", "null") else np.nan)

        X_raw = np.array(vals).reshape(1, -1)

        # Imputasi: gunakan mean dataset jika NaN
        for i, v in enumerate(X_raw[0]):
            if np.isnan(v):
                X_raw[0, i] = float(scaler_mean[i])

        X_scaled = (X_raw - scaler_mean) / scaler_std
        X_3d     = X_scaled.reshape(1, 6, 1)

        prob       = float(model_ppg.predict(X_3d, verbose=0).flatten()[0])
        label      = "Tinggi" if prob >= 0.5 else "Rendah"
        confidence = prob if prob >= 0.5 else (1 - prob)

        # UFR weights dari analisis 4.5 (PPG)
        ufr_ppg = {
            "HF":          50.26,
            "MeanNN_ms":   26.26,
            "LF_HF_Ratio": 17.00,
            "LF":           4.98,
            "SDNN_ms":      1.27,
            "RMSSD_ms":     0.24,
        }

        return safe_json({
            "label":       label,
            "probability": round(prob, 4),
            "confidence":  round(confidence * 100, 1),
            "fitur_input": {f: (round(float(X_raw[0][i]), 3) if not math.isnan(float(X_raw[0][i])) else None)
                            for i, f in enumerate(FITUR)},
            "ufr_weights": ufr_ppg,
        })
    except Exception as e:
        return safe_json({"error": str(e)}), 500


@app.route("/api/process-signal", methods=["POST"])
def api_process_signal():
    """
    Terima file CSV sinyal mentah (ECG atau PPG dari CLAS),
    jalankan pipeline NeuroKit2, kembalikan sinyal + fitur HRV.
    """
    if "file" not in request.files:
        return safe_json({"error": "Tidak ada file yang diunggah"}), 400

    f           = request.files["file"]
    signal_type = request.form.get("signal_type", "ppg").lower()
    SAMPLING_RATE = 256

    try:
        import neurokit2 as nk
        content = f.read().decode("utf-8", errors="replace")
        df_raw  = pd.read_csv(io.StringIO(content))

        # Deteksi kolom sinyal
        if signal_type == "ecg":
            col = "ecg2" if "ecg2" in df_raw.columns else df_raw.select_dtypes(include="number").columns[-1]
        else:
            col = "ppg" if "ppg" in df_raw.columns else df_raw.select_dtypes(include="number").columns[-1]

        df_raw     = df_raw[pd.to_numeric(df_raw[col], errors="coerce").notna()]
        signal_raw = df_raw[col].astype(float).values

        # Batasi 60 detik
        MAX_SAMPLES = SAMPLING_RATE * 60
        signal_raw  = signal_raw[:MAX_SAMPLES]

        # Pipeline NeuroKit2
        if signal_type == "ecg":
            signal_clean = nk.ecg_clean(signal_raw, sampling_rate=SAMPLING_RATE)
            _, info      = nk.ecg_peaks(signal_clean, sampling_rate=SAMPLING_RATE)
            peaks        = info["ECG_R_Peaks"]
        else:
            signal_clean = nk.ppg_clean(signal_raw, sampling_rate=SAMPLING_RATE)
            info_peaks   = nk.ppg_findpeaks(signal_clean, sampling_rate=SAMPLING_RATE)
            peaks        = info_peaks["PPG_Peaks"]

        hrv_time = nk.hrv_time(peaks, sampling_rate=SAMPLING_RATE, show=False)
        hrv_freq = nk.hrv_frequency(peaks, sampling_rate=SAMPLING_RATE, show=False)

        def safe_val(df_row, col_name):
            try:
                v = float(df_row[col_name].values[0])
                return None if (math.isnan(v) or math.isinf(v)) else round(v, 4)
            except Exception:
                return None

        features = {
            "MeanNN_ms":   safe_val(hrv_time, "HRV_MeanNN"),
            "SDNN_ms":     safe_val(hrv_time, "HRV_SDNN"),
            "RMSSD_ms":    safe_val(hrv_time, "HRV_RMSSD"),
            "LF":          safe_val(hrv_freq, "HRV_LF"),
            "HF":          safe_val(hrv_freq, "HRV_HF"),
            "LF_HF_Ratio": safe_val(hrv_freq, "HRV_LFHF"),
        }

        # Downsample sinyal untuk tampilan (max 1024 titik)
        step   = max(1, len(signal_raw) // 1024)
        t_axis = (np.arange(0, len(signal_raw), step) / SAMPLING_RATE).round(3).tolist()
        raw_ds = signal_raw[::step].tolist()
        cln_ds = signal_clean[::step].tolist()

        def norm(arr):
            mn, mx = min(arr), max(arr)
            rng = mx - mn if mx != mn else 1
            return [round((v - mn) / rng, 4) for v in arr]

        peaks_t = (peaks / SAMPLING_RATE).round(3).tolist()

        return safe_json({
            "signal_type":  signal_type,
            "duration_sec": round(len(signal_raw) / SAMPLING_RATE, 1),
            "n_peaks":      int(len(peaks)),
            "n_samples":    int(len(signal_raw)),
            "t_axis":       t_axis,
            "raw":          norm(raw_ds),
            "clean":        norm(cln_ds),
            "peaks_t":      peaks_t,
            "features":     features,
        })

    except Exception as e:
        import traceback
        return safe_json({"error": str(e), "trace": traceback.format_exc()}), 500


# ─── Startup ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_model_and_scaler()
    app.run(debug=True, host="0.0.0.0", port=5000)
