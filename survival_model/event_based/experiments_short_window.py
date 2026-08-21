"""Strategi 'stop kejar C-index, fokus operasional': jendela corrective
SANGAT dekat (7/14 hari) di atas fitur produksi final - hipotesis: sinyal
"baru saja bermasalah" lebih tajam untuk AUC-30d/Recall@kapasitas (horizon
DEKAT) dibanding jendela 30/60/90 hari yang sudah ada.

Floor (TIDAK boleh turun): C-index t0-only baseline. Target (mau naik):
AUC-30d/90d - proxy murah untuk Recall@kapasitas operasional.
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
sys.path.insert(0, str(EVENT_BASED_DIR))

import pandas as pd

import build_dataset
from eb_src import dynamic_history, features
from experiments import fit_eval, render_table

REPORTS_DIR = EVENT_BASED_DIR / "reports"


def main() -> int:
    built = build_dataset.build()
    dataset, base_features = built["dataset"], built["features"]
    landmarks, events = built["landmarks"], built["events"]

    short_window = dynamic_history.windowed_corrective_extra(landmarks, events, windows=(7, 14))

    feature_frame = pd.concat([base_features, short_window], axis=1)
    numeric_cols = features.NUMERIC_FEATURES + features.FLEET_FEATURES + list(short_window.columns)

    print("[kandidat] + jendela corrective 7/14 hari...")
    result = fit_eval(
        "I_plus_short_window_7_14d", feature_frame, features.CATEGORICAL_FEATURES, numeric_cols, dataset,
    )

    report = ["# I_plus_short_window_7_14d: jendela corrective sangat dekat (Fase 3)", ""]
    report.append(
        "Baseline (0_baseline_production) ada di reports/fleet_hierarchy.md (VAL t0=0,7985, "
        "AUC30=0,7862) - dibandingkan di sini tanpa diulang."
    )
    report.append("")
    report.append(render_table(result["rows"]))
    (REPORTS_DIR / "short_window.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\n[OK] Laporan: {REPORTS_DIR / 'short_window.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
