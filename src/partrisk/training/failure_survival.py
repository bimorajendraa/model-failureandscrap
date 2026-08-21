"""Latih model survival event-based (RSF + Cox PH) pada landmark PART.

    python -m partrisk.training.failure_survival

Reuse TOTAL logic fitting/evaluasi native dari `survival.model_fit` (TIDAK
diubah, TIDAK disalin) - satu-satunya yang beda dari model classification
adalah sumber datanya (`training.datasets.survival`, banyak baris/lifecycle
per lifecycle) dan encoder/fitur (`features.survival.builder`, kolom
categorical/numeric berbeda karena umur pemasangan sekarang jadi fitur,
bukan sumbu waktu konstan).

BELUM menulis models/failure/v3/ (mode aditif - lihat gate_decision.md,
model ini TIDAK menggantikan CatBoost). ARTIFACTS_DIR SEMENTARA masih
menunjuk artifact riset (survival_model/event_based/artifacts/) - konfigurasi
di sini MASIH konfigurasi riset penuh (5,26 GB), BELUM konfigurasi compact
pemenang Fase A2 (lihat survival_model/event_based/reports/gate_a2_compact_model.md) -
itu perubahan PERILAKU terpisah, bukan bagian dari pemindahan mekanis ini.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib

from partrisk import config
from partrisk.features.survival import builder as features
from partrisk.survival import model_fit
from partrisk.training.datasets import survival as build_dataset

ARTIFACTS_DIR = config.PACKAGE_DIR / "survival_model" / "event_based" / "artifacts"


def main() -> int:
    print("[1/4] Menyusun dataset event-based (baca database, landmark)...")
    built = build_dataset.build()
    dataset, feature_frame = built["dataset"], built["features"]

    train_mask = (dataset["split"] == "TRAIN").to_numpy()
    val_mask = (dataset["split"] == "VALIDATION").to_numpy()
    test_mask = (dataset["split"] == "TEST").to_numpy()
    print(
        f"      TRAIN={int(train_mask.sum()):,} ({dataset.loc[train_mask,'installation_cycle_id'].nunique():,} lifecycle)  "
        f"VALIDATION={int(val_mask.sum()):,} ({dataset.loc[val_mask,'installation_cycle_id'].nunique():,} lifecycle)  "
        f"TEST={int(test_mask.sum()):,} ({dataset.loc[test_mask,'installation_cycle_id'].nunique():,} lifecycle)"
    )

    print("[2/4] Encoding fitur (one-hot kategorikal, fit di TRAIN saja)...")
    encoder = features.fit_encoder(feature_frame.loc[train_mask])
    x_train = features.encode(feature_frame.loc[train_mask], encoder)
    x_val = features.encode(feature_frame.loc[val_mask], encoder)
    x_test = features.encode(feature_frame.loc[test_mask], encoder)
    y_train = model_fit.make_survival_target(dataset, train_mask)
    y_val = model_fit.make_survival_target(dataset, val_mask)
    y_test = model_fit.make_survival_target(dataset, test_mask)

    print("[3/4] Melatih Random Survival Forest + Cox PH (landmark, banyak baris/lifecycle)...")
    # RSF_PARAMS SAMA dengan model statis (titik awal, BELUM di-tuning ulang
    # khusus populasi landmark - lihat README bagian "Belum dikerjakan").
    models = model_fit.fit_models(x_train, y_train)
    metrics = model_fit.evaluate_models(models, y_train, x_val, y_val, x_test, y_test)
    for name in models:
        print(
            f"      {name:24s} C-index(full landmark) val={metrics[name]['validation']['c_index']:.4f}  "
            f"test={metrics[name]['test']['c_index']:.4f}"
        )

    print("[4/4] Menyimpan artifacts...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, ARTIFACTS_DIR / "models.joblib")
    joblib.dump(encoder, ARTIFACTS_DIR / "encoder.joblib")
    joblib.dump(y_train, ARTIFACTS_DIR / "y_train.joblib")

    metadata = {
        "training_date": datetime.now(timezone.utc).isoformat(),
        "data_end": str(built["data_end"]),
        "primary_model": "random_survival_forest",
        "unit_of_observation": (
            "one row per (lifecycle, landmark) - features/age RE-ANCHORED at each landmark's "
            "observation_on (install / organic operational event / sparse 90-365d anchor), "
            "NOT frozen at installed_on like a baseline-installation model"
        ),
        "target": "duration_days (residual time from landmark's observation_on to failure/censoring), event_observed",
        "landmark_design": {
            "sources": ["INSTALL (age=0, always)", "ORGANIC_EVENT (operational event mid-cycle)", "ANCHOR (90/180/365d then +365d, capped)"],
            "split_assignment": "follows the LIFECYCLE's installed_on (NOT per-landmark L) - see features/survival/landmarks.py docstring",
        },
        "feature_columns": features.FEATURE_COLUMNS,
        "categorical_features": features.CATEGORICAL_FEATURES,
        "category_thresholds": features.FINAL_CATEGORY_THRESHOLDS,
        "rows_by_split": dataset["split"].value_counts().to_dict(),
        "lifecycles_by_split": dataset.groupby("split")["installation_cycle_id"].nunique().to_dict(),
        "events_by_split": dataset.groupby("split")["event_observed"].sum().to_dict(),
        "support_totals": built["support_totals"],
        "item_type_support_totals": built["item_type_support_totals"],
        "terminal_support_totals": built["terminal_support_totals"],
        "hyperparameters": {
            "random_survival_forest": model_fit.DEFAULT_RSF_PARAMS,
            "cox_ph": model_fit.DEFAULT_COX_PARAMS,
        },
        "evaluation_metrics_full_landmark_rows": metrics,
    }
    (ARTIFACTS_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"      Tersimpan di {ARTIFACTS_DIR}")
    print()
    print("      PERINGATAN: C-index di atas dihitung dari SEMUA baris landmark (repeated")
    print("      measures per lifecycle - BUKAN apples-to-apples dengan populasi t0-only).")
    print("      Jalankan scripts/evaluate_survival.py untuk perbandingan t0-only yang adil.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
