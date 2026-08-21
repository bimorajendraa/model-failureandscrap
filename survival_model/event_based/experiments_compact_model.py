"""Fase A2 (plan restrukturisasi): cari konfigurasi RSF event-based yang
artifact-nya <=100 MB tanpa menurunkan C-index VALIDATION secara berarti.

Artifact produksi sekarang (`artifacts/models.joblib`) 5,26 GB. Ukurannya
kira-kira `n_trees x n_leaves x |unique event times|` (README "Catatan
teknis"). Sudah diukur: dari 3.057 unique event time di TRAIN, cuma 120
(<=120 hari) yang benar-benar dipakai kontrak API (risk_30/60/90/120d) - 96,1%
grid tidak terpakai.

Lever di sini, TANPA matematika baru:
  1. Perkasar target `duration_days` yang dilihat RSF.fit() (BUKAN yang
     dipakai evaluasi) - resolusi harian s/d 120 hari (kontrak API), lalu
     kelipatan 30 hari di atasnya (median survival time tetap butuh grid
     panjang, jadi TIDAK dipotong, cuma dikasarkan).
  2. n_estimators 100->60, min_samples_leaf 30->80 (leaf lebih besar = lebih
     sedikit leaf = lebih sedikit salinan grid waktu).

Evaluasi TETAP pakai `duration_days` ASLI (tidak dikasarkan) untuk y_train
(distribusi censoring buat Uno's C-index) dan y_val - mengasarkan HANYA
target yang dilihat `.fit()`, supaya angka VALIDATION di sini bisa
dibandingkan apa adanya dengan `artifacts/metadata.json` sekarang (RSF VAL
c_index=0,8290, ibs=0,04747, brier@30=0,03570, auc@30=0,8357).

Keputusan HANYA dari VALIDATION (aturan proyek, bukan TEST) - lihat
`render()` di bawah untuk kriteria lulus.

    python survival_model/event_based/experiments_compact_model.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SURVIVAL_DIR = Path(__file__).resolve().parent.parent
if str(SURVIVAL_DIR) not in sys.path:
    sys.path.insert(0, str(SURVIVAL_DIR))
EVENT_BASED_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVENT_BASED_DIR))

import joblib
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

import build_dataset
from src import evaluation, model_fit

from eb_src import features

REPORTS_DIR = EVENT_BASED_DIR / "reports"

# Baseline produksi sekarang (artifacts/metadata.json, RSF, VALIDATION,
# full-landmark rows) - dicatat di sini sebagai angka pembanding statis
# supaya skrip ini tidak perlu me-load ulang artifact 5,26 GB cuma untuk
# baca metadata-nya.
BASELINE_VAL = {
    "c_index": 0.8290270977967045,
    "integrated_brier_score": 0.04746517713658578,
    "brier_at_30": 0.035696829938949154,
    "auc_at_30": 0.8356710898315958,
}
# Toleransi turun (bukan gerbang G1-G4 - itu soal kalah/menang lawan
# CatBoost, sudah diputuskan GAGAL di A1. Ini cuma "jangan buang akurasi
# secara sia-sia" waktu mengecilkan artifact untuk mode aditif).
C_INDEX_FLOOR = BASELINE_VAL["c_index"] - 0.01

COMPACT_RSF_PARAMS = dict(
    n_estimators=50,
    min_samples_split=140,
    min_samples_leaf=100,
    max_features="sqrt",
    n_jobs=1,  # lihat catatan evaluate.py soal loky hang setelah unpickle
    random_state=42,
)


def coarsen_duration_days(days: np.ndarray) -> np.ndarray:
    """Resolusi harian s/d 120 hari (horizon kontrak API), kelipatan 60 hari
    di atasnya (percobaan pertama pakai kelipatan 30 -> 126,6 MB, masih di
    atas ambang G5 100 MB - lihat reports/gate_a2_compact_model.md riwayat
    percobaan pertama). HANYA dipakai untuk target yang dilihat RSF.fit() -
    evaluasi tetap pakai duration_days asli (lihat docstring modul)."""
    days = np.asarray(days, dtype=float)
    near = np.round(days)
    far = 120.0 + np.round((days - 120.0) / 60.0) * 60.0
    return np.maximum(np.where(days <= 120.0, near, far), 1.0)


def _artifact_size_mb(model) -> float:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.joblib"
        joblib.dump(model, path)
        return path.stat().st_size / 1e6


def main() -> int:
    print("[1/3] Menyusun dataset event-based (cache lokal kalau ada - HANYA untuk pencarian hyperparameter;")
    print("      konfigurasi terpilih WAJIB dilatih ulang dari DB fresh sebelum jadi artifact produksi, lihat G8)...")
    import os

    os.environ.setdefault("SURVIVAL_BUILD_CACHE", "1")
    built = build_dataset.build()
    dataset, feature_frame = built["dataset"], built["features"]

    train_mask = (dataset["split"] == "TRAIN").to_numpy()
    val_mask = (dataset["split"] == "VALIDATION").to_numpy()

    print("[2/3] Encoding fitur (sama seperti train.py)...")
    encoder = features.fit_encoder(feature_frame.loc[train_mask])
    x_train = features.encode(feature_frame.loc[train_mask], encoder)
    x_val = features.encode(feature_frame.loc[val_mask], encoder)

    y_train_true = model_fit.make_survival_target(dataset, train_mask)
    y_val = model_fit.make_survival_target(dataset, val_mask)

    y_train_coarse = Surv.from_arrays(
        event=y_train_true["event"],
        time=coarsen_duration_days(y_train_true["time"]),
    )
    n_unique_true = len(np.unique(y_train_true["time"][y_train_true["event"]]))
    n_unique_coarse = len(np.unique(y_train_coarse["time"][y_train_coarse["event"]]))
    print(f"      unique event times TRAIN: {n_unique_true:,} -> {n_unique_coarse:,} sesudah dikasarkan")

    print("[3/3] Melatih kandidat compact + mengukur ukuran artifact riil (serialize joblib)...")
    t0 = time.time()
    model = RandomSurvivalForest(**COMPACT_RSF_PARAMS).fit(x_train, y_train_coarse)
    fit_s = time.time() - t0
    size_mb = _artifact_size_mb(model)

    # Evaluasi pakai duration_days ASLI (y_train_true, y_val) - konsisten
    # dengan cara model production dievaluasi (model_fit.evaluate_models).
    metrics = evaluation.native_metrics(model, y_train_true, x_val, y_val, risk_sign=1)

    rows = [
        ("baseline (production, 5.26 GB)", "n_estimators=100 min_samples_leaf=30 grid asli",
         5262.3, BASELINE_VAL["c_index"], BASELINE_VAL["integrated_brier_score"],
         BASELINE_VAL["brier_at_30"], BASELINE_VAL["auc_at_30"]),
        (
            "compact (kandidat A2)",
            f"n_estimators={COMPACT_RSF_PARAMS['n_estimators']} "
            f"min_samples_leaf={COMPACT_RSF_PARAMS['min_samples_leaf']} "
            f"grid dikasarkan ({n_unique_coarse} titik)",
            size_mb, metrics["c_index"], metrics["integrated_brier_score"],
            metrics["brier_at_horizon"][30], metrics["time_dependent_auc_at_horizon"][30],
        ),
    ]

    lines = [
        "| Konfigurasi | Detail | Ukuran (MB) | VAL C-index | VAL IBS | VAL Brier@30 | VAL AUC@30 |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, detail, mb, c_idx, ibs, brier30, auc30 in rows:
        lines.append(f"| {label} | {detail} | {mb:,.1f} | {c_idx:.4f} | {ibs:.4f} | {brier30:.4f} | {auc30:.4f} |")
    table = "\n".join(lines)
    print()
    print(table)
    print()
    print(f"      Fit compact: {fit_s:.1f} detik")

    passed = size_mb <= 100.0 and metrics["c_index"] >= C_INDEX_FLOOR
    verdict = (
        "LULUS - artifact <=100 MB, C-index VALIDATION dalam toleransi (>= baseline - 0,01)."
        if passed else
        "GAGAL - lihat angka di atas (ukuran masih >100 MB, atau C-index turun lebih dari 0,01)."
    )
    print(f"      {verdict}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = f"""# Fase A2: pencarian model compact (event-based RSF)

Populasi: VALIDATION landmark rows (aturan proyek - keputusan tidak pernah dari TEST).
Konfigurasi TRAIN dari `SURVIVAL_BUILD_CACHE` (pencarian cepat) - konfigurasi terpilih
WAJIB dilatih ulang dari pembacaan DB fresh sebelum jadi artifact produksi (G8).

{table}

Fit compact: {fit_s:.1f} detik.

Verdict: {verdict}

Lever dipakai: perkasar target duration_days yang dilihat RSF.fit() (resolusi harian
s/d 120 hari, kelipatan 30 hari di atasnya - evaluasi tetap pakai duration_days ASLI),
n_estimators 100->60, min_samples_leaf 30->80, min_samples_split 40->110 (rasio dijaga
mirip baseline).
"""
    (REPORTS_DIR / "gate_a2_compact_model.md").write_text(report, encoding="utf-8")
    print(f"\n[OK] Laporan: {REPORTS_DIR / 'gate_a2_compact_model.md'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
