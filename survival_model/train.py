"""Latih model survival (Random Survival Forest + Cox PH) pada lifecycle PART.

    python survival_model/train.py

Alurnya:

    build_dataset.build() -> encode kategorikal (fit di TRAIN) -> latih RSF + CoxPH
    -> evaluasi native (C-index dll, lewat evaluate.py) -> simpan artifacts/

Dua model dilatih dan dilaporkan berdampingan (pola yang sama seperti
perbandingan LogReg+RF di train_scrap.py) - RSF adalah model utama yang
disimpan sebagai "primary", CoxPH baseline sederhana untuk pembanding. Tidak
ada pencarian hyperparameter besar-besaran; tujuannya membuktikan formulasi
survival, bukan memeras skor tertinggi.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SURVIVAL_DIR = Path(__file__).resolve().parent
if str(SURVIVAL_DIR) not in sys.path:
    sys.path.insert(0, str(SURVIVAL_DIR))

import joblib
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

import build_dataset
from src import evaluation, features

ARTIFACTS_DIR = SURVIVAL_DIR / "artifacts"

RSF_PARAMS = dict(
    n_estimators=100,
    min_samples_split=40,
    min_samples_leaf=30,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
    # low_memory=True TIDAK dipakai - itu mematikan predict_survival_function()
    # sepenuhnya, yang justru inti eksperimen ini. Ukuran artifact yang wajar
    # dicapai lewat pembulatan duration_days ke hari bulat (lihat
    # src/lifecycle_builder.py) - itu yang tadinya membuat artifact >4 GiB
    # (grid waktu unik meledak sampai ribuan titik gara-gara presisi jam/menit
    # timestamp mentah), bukan n_estimators/min_samples_leaf di sini.
)
COX_PARAMS = dict(alpha=0.1, ties="efron")


def make_survival_target(dataset, mask):
    return Surv.from_arrays(
        event=dataset.loc[mask, "event_observed"].astype(bool).to_numpy(),
        time=dataset.loc[mask, "duration_days"].to_numpy(),
    )


def main() -> int:
    print("[1/4] Menyusun dataset survival (baca database)...")
    built = build_dataset.build()
    dataset, feature_frame = built["dataset"], built["features"]

    train_mask = (dataset["split"] == "TRAIN").to_numpy()
    val_mask = (dataset["split"] == "VALIDATION").to_numpy()
    test_mask = (dataset["split"] == "TEST").to_numpy()
    print(
        f"      TRAIN={int(train_mask.sum()):,}  VALIDATION={int(val_mask.sum()):,}  "
        f"TEST={int(test_mask.sum()):,}"
    )

    print("[2/4] Encoding fitur (one-hot kategorikal, fit di TRAIN saja)...")
    encoder = features.fit_encoder(feature_frame.loc[train_mask])
    x_train = features.encode(feature_frame.loc[train_mask], encoder)
    x_val = features.encode(feature_frame.loc[val_mask], encoder)
    x_test = features.encode(feature_frame.loc[test_mask], encoder)
    y_train = make_survival_target(dataset, train_mask)
    y_val = make_survival_target(dataset, val_mask)
    y_test = make_survival_target(dataset, test_mask)

    print("[3/4] Melatih Random Survival Forest + Cox PH...")
    rsf = RandomSurvivalForest(**RSF_PARAMS).fit(x_train, y_train)
    cox = CoxPHSurvivalAnalysis(**COX_PARAMS).fit(x_train, y_train)

    models = {"random_survival_forest": rsf, "cox_ph": cox}
    metrics = {}
    for name, model in models.items():
        metrics[name] = {
            "validation": evaluation.native_metrics(model, y_train, x_val, y_val),
            "test": evaluation.native_metrics(model, y_train, x_test, y_test),
        }
        print(
            f"      {name:24s} C-index val={metrics[name]['validation']['c_index']:.4f}  "
            f"test={metrics[name]['test']['c_index']:.4f}  "
            f"IBS val={metrics[name]['validation']['integrated_brier_score']}  "
            f"test={metrics[name]['test']['integrated_brier_score']}"
        )

    print("[4/4] Menyimpan artifacts...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, ARTIFACTS_DIR / "models.joblib")
    joblib.dump(encoder, ARTIFACTS_DIR / "encoder.joblib")
    joblib.dump(y_train, ARTIFACTS_DIR / "y_train.joblib")  # dibutuhkan IPCW saat evaluasi/inference ulang

    metadata = {
        "training_date": datetime.now(timezone.utc).isoformat(),
        "data_end": str(built["data_end"]),
        "primary_model": "random_survival_forest",
        "unit_of_observation": "one row per installation lifecycle (episode), "
        "features anchored at installed_on (cycle start) - NOT current PART condition",
        "target": "duration_days (time from installed_on to failure/censoring), event_observed (1=failure, 0=censored)",
        "feature_columns": features.FEATURE_COLUMNS,
        "categorical_features": features.CATEGORICAL_FEATURES,
        "dropped_from_classification_features": features.DROPPED_AT_INSTALL_FEATURES,
        "rows_by_split": dataset["split"].value_counts().to_dict(),
        "events_by_split": dataset.groupby("split")["event_observed"].sum().to_dict(),
        "support_totals": built["support_totals"],
        "hyperparameters": {"random_survival_forest": RSF_PARAMS, "cox_ph": COX_PARAMS},
        "metrics": metrics,
    }
    (ARTIFACTS_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"      Tersimpan di {ARTIFACTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
