"""Concept drift: apakah TRAIN dari jendela tahun LEBIH BARU (bukan seluruh
2014-2024) memberi VAL t0-only C-index lebih baik, pada fitur PRODUKSI
final saat ini (G_combined_without_device - lihat eb_src/features.py) -
pola maintenance/device bisa berubah, lebih banyak data lama tidak otomatis
lebih baik.

Split assignment TIDAK diubah (VALIDATION/TEST tetap identik dengan
train.py) - hanya BARIS TRAIN yang dipangkas berdasar installed_on
lifecycle-nya (bukan observation_on landmark), supaya satu lifecycle tetap
tidak pernah terpotong di tengah.

    python survival_model/event_based/concept_drift.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SURVIVAL_DIR = Path(__file__).resolve().parent.parent
if str(SURVIVAL_DIR) not in sys.path:
    sys.path.insert(0, str(SURVIVAL_DIR))
EVENT_BASED_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVENT_BASED_DIR))  # lihat catatan build_dataset.py/train.py

import pandas as pd

from src import evaluation, model_fit

import build_dataset
from eb_src import features
from experiments import render_table

REPORTS_DIR = EVENT_BASED_DIR / "reports"
TRAIN_WINDOWS = [("2014-2024 (penuh)", "2014-01-01"), ("2018-2024", "2018-01-01"), ("2020-2024", "2020-01-01"), ("2022-2024", "2022-01-01")]


def main() -> int:
    print("[1/2] Memuat dataset event-based (fitur PRODUKSI final saat ini)...")
    built = build_dataset.build()
    dataset, feature_frame = built["dataset"], built["features"]

    val_mask = (dataset["split"] == "VALIDATION").to_numpy()
    t0_mask = (dataset["landmark_source"] == "INSTALL").to_numpy()
    val_t0_mask = val_mask & t0_mask
    installed_on = pd.to_datetime(dataset["installed_on"])

    print("[2/2] Fit RSF per jendela TRAIN, evaluasi VAL t0-only (SEBANDING antar jendela)...")
    rows = []
    for label, cutoff in TRAIN_WINDOWS:
        train_mask = (dataset["split"] == "TRAIN").to_numpy() & (installed_on >= pd.Timestamp(cutoff)).to_numpy()
        n_train_lifecycles = dataset.loc[train_mask, "installation_cycle_id"].nunique()

        encoder = features.fit_encoder(feature_frame.loc[train_mask])
        x_train = features.encode(feature_frame.loc[train_mask], encoder)
        x_val_t0 = features.encode(feature_frame.loc[val_t0_mask], encoder)
        y_train = model_fit.make_survival_target(dataset, train_mask)
        y_val_t0 = model_fit.make_survival_target(dataset, val_t0_mask)

        rsf_params = {**model_fit.DEFAULT_RSF_PARAMS, "n_jobs": 1}
        models = model_fit.fit_models(x_train, y_train, ["random_survival_forest"], {"random_survival_forest": rsf_params})
        m = evaluation.native_metrics(models["random_survival_forest"], y_train, x_val_t0, y_val_t0)
        rows.append({
            "label": f"{label} ({n_train_lifecycles:,} lifecycle)", "model": "random_survival_forest",
            "val_full_c_index": m["c_index"], "val_t0_c_index": m["c_index"], "val_t0_ibs": m["integrated_brier_score"],
        })
        print(f"      {label:22s} n_lifecycle={n_train_lifecycles:6,}  VAL-t0={m['c_index']:.4f}")

    report = ["# Concept drift: jendela tahun TRAIN (event-based, fitur produksi final)", ""]
    report.append(
        "TRAIN dipangkas berdasar `installed_on` LIFECYCLE (bukan observation_on landmark) - satu lifecycle "
        "tidak pernah terpotong di tengah. VALIDATION identik di semua baris (t0-only, sama seperti "
        "reports/evaluation_report.md)."
    )
    report.append("")
    report.append(render_table(rows))
    (REPORTS_DIR / "concept_drift.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\n[OK] Laporan: {REPORTS_DIR / 'concept_drift.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
