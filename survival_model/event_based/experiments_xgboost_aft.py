"""Uji XGBoost AFT (Accelerated Failure Time) pada fitur PRODUKSI final saat
ini - dibatasi waktu ketat (XGBoost histogram-based biasanya JAUH lebih
cepat dari sksurv GBSA yang sebelumnya dihentikan setelah 76 menit tanpa
hasil). Evaluasi VAL t0-only (SEBANDING dengan model statis).

xgboost bukan dependency permanen project ini - HANYA dipasang untuk
eksperimen ini (lihat requirements.txt tidak diubah). Kalau AFT terbukti
membantu, baru dipertimbangkan jadi dependency resmi.
"""

from __future__ import annotations

import sys
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

import numpy as np
import xgboost as xgb
from sksurv.metrics import concordance_index_censored

from src import model_fit

import build_dataset
from eb_src import features

REPORTS_DIR = EVENT_BASED_DIR / "reports"


def main() -> int:
    print("[1/3] Memuat dataset (fitur produksi final)...")
    built = build_dataset.build()
    dataset, feature_frame = built["dataset"], built["features"]

    train_mask = (dataset["split"] == "TRAIN").to_numpy()
    val_mask = (dataset["split"] == "VALIDATION").to_numpy()
    t0_mask = (dataset["landmark_source"] == "INSTALL").to_numpy()
    val_t0_mask = val_mask & t0_mask

    encoder = features.fit_encoder(feature_frame.loc[train_mask])
    x_train = features.encode(feature_frame.loc[train_mask], encoder)
    x_val_t0 = features.encode(feature_frame.loc[val_t0_mask], encoder)

    duration_train = dataset.loc[train_mask, "duration_days"].to_numpy()
    event_train = dataset.loc[train_mask, "event_observed"].to_numpy().astype(bool)
    # AFT butuh label_lower_bound/label_upper_bound (BUKAN satu kolom target):
    # event=1 (failure pasti terjadi PERSIS di durasi ini) -> lower=upper=duration.
    # event=0 (censored - HANYA tahu bertahan SAMPAI durasi ini, bisa lebih lama)
    # -> lower=duration, upper=+inf.
    y_lower = duration_train.astype(float)
    y_upper = np.where(event_train, duration_train, np.inf).astype(float)

    dtrain = xgb.DMatrix(x_train.to_numpy(dtype=float))
    dtrain.set_float_info("label_lower_bound", y_lower)
    dtrain.set_float_info("label_upper_bound", y_upper)
    dval = xgb.DMatrix(x_val_t0.to_numpy(dtype=float))

    params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": 1.2,
        "max_depth": 4,
        "learning_rate": 0.05,
        "tree_method": "hist",
    }

    print("[2/3] Melatih XGBoost AFT (n_estimators=200, dibatasi waktu)...")
    t0 = time.time()
    booster = xgb.train(params, dtrain, num_boost_round=200)
    fit_time = time.time() - t0
    print(f"      Selesai fit dalam {fit_time:.1f} detik")

    print("[3/3] Evaluasi VAL t0-only...")
    predicted_time = booster.predict(dval)
    # AFT memprediksi WAKTU (bukan skor risiko) - risk = -predicted_time
    # (waktu bertahan lebih pendek -> risiko lebih tinggi), sama seperti
    # peringatan risk_sign di src/model_fit.py untuk GBSA loss='ipcwls'.
    risk = -predicted_time

    y_val_t0 = model_fit.make_survival_target(dataset, val_t0_mask)
    c_index = concordance_index_censored(y_val_t0["event"], y_val_t0["time"], risk)[0]
    print(f"      XGBoost AFT VAL-t0 C-index = {c_index:.4f}")

    report = ["# XGBoost AFT pada fitur produksi final (Fase 3)", ""]
    report.append(f"Fit time: {fit_time:.1f} detik (n_estimators=200, max_depth=4).")
    report.append("")
    report.append("| Model | VAL t0-only C-index |")
    report.append("|---|---|")
    report.append(f"| XGBoost AFT (normal, scale=1.2) | {c_index:.4f} |")
    report.append("| RSF (produksi saat ini) | 0.7985 |")
    report.append("| Cox PH (produksi saat ini) | 0.7651 |")
    (REPORTS_DIR / "xgboost_aft.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\n[OK] Laporan: {REPORTS_DIR / 'xgboost_aft.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
