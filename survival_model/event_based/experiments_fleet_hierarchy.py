"""Uji fitur fleet hierarchy (level item_type, bukan item_model_code) di
atas fitur PRODUKSI final saat ini - kandidat baru dari saran eksternal,
diverifikasi lewat VAL t0-only sebelum diputuskan diikutkan atau tidak."""

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
sys.path.insert(0, str(EVENT_BASED_DIR))

import pandas as pd

from src import install_context

import build_dataset
from eb_src import dynamic_history, features
from experiments import fit_eval, render_table

REPORTS_DIR = EVENT_BASED_DIR / "reports"


def main() -> int:
    built = build_dataset.build()
    dataset, base_features = built["dataset"], built["features"]
    landmarks, cycles, events, episodes = built["landmarks"], built["cycles"], built["events"], built["episodes"]

    cycles_with_type = install_context.attach_install_context(cycles, events)
    fh = dynamic_history.fleet_hierarchy_features(landmarks, cycles_with_type, episodes)

    feature_frame = pd.concat([base_features, fh], axis=1)
    categorical_cols = features.CATEGORICAL_FEATURES
    numeric_cols = features.NUMERIC_FEATURES + features.FLEET_FEATURES + list(fh.columns)

    print("[baseline] fitur produksi final (floor - TIDAK boleh turun)...")
    baseline_result = fit_eval(
        "0_baseline_production", base_features, features.CATEGORICAL_FEATURES,
        features.NUMERIC_FEATURES + features.FLEET_FEATURES, dataset,
    )
    print("[kandidat] + fleet hierarchy...")
    result = fit_eval("H_plus_fleet_hierarchy", feature_frame, categorical_cols, numeric_cols, dataset)

    report = ["# H_plus_fleet_hierarchy: fleet failure rate level item_type (Fase 3)", ""]
    report.append(
        "**Strategi baru**: C-index TIDAK dikejar lagi (sudah terbukti mentok ~0,80 lewat "
        "banyak percobaan) - baris `0_baseline_production` adalah PAGAR (floor), kandidat "
        "HANYA layak diadopsi kalau C-index-nya TIDAK TURUN dari baseline DAN AUC-30d/90d "
        "(proxy murah untuk Recall@kapasitas operasional, tidak perlu bangun ulang populasi "
        "TEST classification 1,4 juta baris) NAIK."
    )
    report.append("")
    report.append(render_table(baseline_result["rows"] + result["rows"]))
    (REPORTS_DIR / "fleet_hierarchy.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\n[OK] Laporan: {REPORTS_DIR / 'fleet_hierarchy.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
