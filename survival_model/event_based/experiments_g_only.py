"""Jalankan HANYA konfigurasi G_combined_without_device (lihat experiments.py)
- cek apakah fitur produksi final BISA menghindari dependency ke schema
`analytics` sama sekali, tanpa kehilangan banyak dari VAL t0-only F_combined_all
(0,8036)."""

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

import build_dataset
from experiments import build_configs, fit_eval, render_table

REPORTS_DIR = EVENT_BASED_DIR / "reports"


def main() -> int:
    built = build_dataset.build()
    dataset = built["dataset"]
    configs = build_configs(built)
    cfg = configs["G_combined_without_device"]
    result = fit_eval(
        "G_combined_without_device", cfg["feature_frame"], cfg["categorical_cols"], cfg["numeric_cols"], dataset,
    )
    report = ["# G_combined_without_device (tanpa dependency schema analytics)", ""]
    report.append("Bandingkan dengan F_combined_all (VAL t0-only RSF=0,8036) di reports/dynamic_ablation.md.")
    report.append("")
    report.append(render_table(result["rows"]))
    (REPORTS_DIR / "g_without_device.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\n[OK] Laporan: {REPORTS_DIR / 'g_without_device.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
