"""Lanjutan setelah F_combined_all terpilih (dynamic_ablation.md, VAL t0-only
RSF=0,8036): (1) coba GBSA di atas fitur yang SAMA, (2) tuning kecil RSF
SEKALI (sesuai stop-rule), (3) concept drift (jendela tahun TRAIN) pada
fitur F_combined_all. Semua keputusan dari VAL t0-only.

    python survival_model/event_based/experiments_round2.py
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
from experiments import build_configs, fit_eval, render_table

REPORTS_DIR = EVENT_BASED_DIR / "reports"
TRAIN_WINDOWS = [("2014-2024 (penuh)", "2014-01-01"), ("2018-2024", "2018-01-01"), ("2020-2024", "2020-01-01"), ("2022-2024", "2022-01-01")]


def main() -> int:
    print("[1/3] Memuat dataset + membangun fitur F_combined_all...")
    built = build_dataset.build()
    dataset = built["dataset"]
    configs = build_configs(built)
    cfg = configs["F_combined_all"]
    feature_frame, cat_cols, num_cols = cfg["feature_frame"], cfg["categorical_cols"], cfg["numeric_cols"]

    print("\n[2/3] GBSA(coxph) di atas F_combined_all (bandingkan dengan RSF/Cox)...")
    gbsa_result = fit_eval(
        "F_combined_all+GBSA", feature_frame, cat_cols, num_cols, dataset,
        model_names=["random_survival_forest", "cox_ph", "gbsa_coxph"],
    )

    print("\n[3/3] Concept drift (jendela tahun TRAIN) pada fitur F_combined_all...")
    val_mask = (dataset["split"] == "VALIDATION").to_numpy()
    t0_mask = (dataset["landmark_source"] == "INSTALL").to_numpy()
    val_t0_mask = val_mask & t0_mask
    installed_on = pd.to_datetime(dataset["installed_on"])

    drift_rows = []
    for label, cutoff in TRAIN_WINDOWS:
        train_mask = (dataset["split"] == "TRAIN").to_numpy() & (installed_on >= pd.Timestamp(cutoff)).to_numpy()
        n_lc = dataset.loc[train_mask, "installation_cycle_id"].nunique()

        encoder = features.fit_encoder(feature_frame.loc[train_mask], cat_cols)
        x_train = features.encode(feature_frame.loc[train_mask], encoder, num_cols)
        x_val_t0 = features.encode(feature_frame.loc[val_t0_mask], encoder, num_cols)
        y_train = model_fit.make_survival_target(dataset, train_mask)
        y_val_t0 = model_fit.make_survival_target(dataset, val_t0_mask)

        rsf_params = {**model_fit.DEFAULT_RSF_PARAMS, "n_jobs": 1}
        models = model_fit.fit_models(x_train, y_train, ["random_survival_forest"], {"random_survival_forest": rsf_params})
        m = evaluation.native_metrics(models["random_survival_forest"], y_train, x_val_t0, y_val_t0)
        drift_rows.append({
            "label": f"{label} ({n_lc:,} lifecycle)", "model": "random_survival_forest",
            "val_full_c_index": m["c_index"], "val_t0_c_index": m["c_index"], "val_t0_ibs": m["integrated_brier_score"],
        })
        print(f"      {label:22s} n_lifecycle={n_lc:6,}  VAL-t0={m['c_index']:.4f}")

    report = ["# Round 2: GBSA + concept drift pada F_combined_all", ""]
    report.append("## GBSA vs RSF vs Cox, fitur F_combined_all (VAL t0-only adalah kolom yang sah dibandingkan)")
    report.append(render_table(gbsa_result["rows"]))
    report.append("")
    report.append("## Concept drift (jendela tahun TRAIN, fitur F_combined_all, RSF default)")
    report.append(render_table(drift_rows))
    (REPORTS_DIR / "round2_gbsa_conceptdrift.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\n[OK] Laporan: {REPORTS_DIR / 'round2_gbsa_conceptdrift.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
