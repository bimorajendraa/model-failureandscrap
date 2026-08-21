"""Ensemble: gabungkan skor risk_30d model STATIS + EVENT-BASED pada
populasi TEST classification yang SAMA - dua model ini terbukti menangkap
sinyal BERBEDA (statis lebih unggul C-index global, event-based lebih
unggul Recall@kapasitas operasional - lihat README masing-masing), jadi
kandidat kuat untuk digabung, BUKAN diulang sinyal yang sama.

Model produksi TIDAK diubah oleh skrip ini - HANYA membaca artifacts kedua
model (survival_model/artifacts/, survival_model/event_based/artifacts/)
dan menghitung metrik ensemble sebagai perbandingan.

Catatan teknis penting: `build_dataset.py`/`evaluate.py` PUNYA NAMA SAMA
di `survival_model/` (statis) dan `survival_model/event_based/` - memuat
KEDUANYA di satu proses Python TIDAK BISA lewat `import` biasa berturutan
(sys.modules meng-cache berdasar nama string 'build_dataset'/'evaluate',
pemuatan kedua akan diam-diam memakai cache modul PERTAMA). Modul ini
membersihkan sys.modules eksplisit di antara dua konteks - lihat
`_load_static()`/`_load_event_based()` di bawah.

    python survival_model/event_based/ensemble_operational.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SURVIVAL_DIR = Path(__file__).resolve().parent.parent
EVENT_BASED_DIR = Path(__file__).resolve().parent

_COLLIDING_MODULE_NAMES = ["build_dataset", "evaluate", "features"]


def _reset_colliding_modules() -> None:
    for name in _COLLIDING_MODULE_NAMES:
        sys.modules.pop(name, None)


def _load_static():
    """sys.path = [ROOT_DIR, SURVIVAL_DIR] SAJA (event_based TIDAK ada di
    path) - `import build_dataset`/`import evaluate` di sini dijamin
    resolve ke versi statis."""
    sys.path[:] = [p for p in sys.path if p not in (str(SURVIVAL_DIR), str(EVENT_BASED_DIR), str(ROOT_DIR))]
    sys.path.insert(0, str(SURVIVAL_DIR))
    sys.path.insert(0, str(ROOT_DIR))
    _reset_colliding_modules()
    import build_dataset
    import evaluate

    return build_dataset, evaluate


def _load_event_based():
    """sys.path = [EVENT_BASED_DIR, SURVIVAL_DIR, ROOT_DIR] - EVENT_BASED_DIR
    PALING DEPAN, `import build_dataset`/`import evaluate`/`import features`
    dijamin resolve ke versi event-based (bukan versi statis yang barusan
    di-cache)."""
    sys.path[:] = [p for p in sys.path if p not in (str(SURVIVAL_DIR), str(EVENT_BASED_DIR), str(ROOT_DIR))]
    sys.path.insert(0, str(EVENT_BASED_DIR))
    sys.path.insert(0, str(SURVIVAL_DIR))
    sys.path.insert(0, str(ROOT_DIR))
    sys.path.insert(0, str(EVENT_BASED_DIR))  # pastikan EVENT_BASED_DIR paling depan
    _reset_colliding_modules()
    import build_dataset
    import evaluate

    return build_dataset, evaluate


def main() -> int:
    print("[1/4] Memuat modul model STATIS (build_dataset + evaluate)...")
    static_build_dataset, static_evaluate = _load_static()
    static_artifacts = static_evaluate.load_artifacts()
    static_model = static_artifacts["models"][static_artifacts["metadata"]["primary_model"]]
    static_built = static_build_dataset.build()
    test_rows, window_days = static_evaluate.load_classification_test_rows()
    print(f"      {len(test_rows):,} baris TEST classification (window {window_days:.0f} hari)")

    static_features = static_built["features"].copy()
    static_features.index = static_built["dataset"]["installation_cycle_id"].to_numpy()

    print("      Menghitung risk_30d model statis...")
    static_computed = static_evaluate.compute_risk_30d(static_model, static_features, static_artifacts["encoder"], test_rows)
    if static_computed is None:
        print("      Model statis tidak punya baris yang cocok - berhenti.")
        return 1
    static_rows, static_risk, static_target = static_computed
    static_df = static_rows[["installation_cycle_id"]].copy()
    static_df["row_key"] = static_rows.index.to_numpy()
    static_df["risk_static"] = static_risk
    static_df["target"] = static_target

    print("\n[2/4] Memuat modul model EVENT-BASED (build_dataset + evaluate)...")
    eb_build_dataset, eb_evaluate = _load_event_based()
    eb_artifacts = eb_evaluate.load_artifacts()
    eb_model = eb_artifacts["models"][eb_artifacts["metadata"]["primary_model"]]
    eb_built = eb_build_dataset.build()

    from eb_src import features as eb_features  # resolve via sys.path saat ini (EVENT_BASED_DIR paling depan)

    eb_dataset, eb_feature_frame = eb_built["dataset"], eb_built["features"]
    t0_mask = (eb_dataset["landmark_source"] == "INSTALL").to_numpy()
    eb_t0_features = eb_feature_frame.loc[t0_mask].copy()
    eb_t0_features.index = eb_dataset.loc[t0_mask, "installation_cycle_id"].to_numpy()
    eb_numeric_columns = eb_features.NUMERIC_FEATURES + eb_features.FLEET_FEATURES

    print("      Menghitung risk_30d model event-based...")
    # Pakai `static_evaluate.compute_risk_30d` yang SUDAH dimuat di [1/4] -
    # fungsi itu generik (terima model/fitur/encoder apa pun sebagai
    # argumen biasa), TIDAK perlu muat ulang lewat eb_evaluate (yang
    # rantai pemuatannya sendiri rawan collision sys.path bersarang).
    eb_computed = static_evaluate.compute_risk_30d(
        eb_model, eb_t0_features, eb_artifacts["encoder"], test_rows, numeric_columns=eb_numeric_columns
    )
    if eb_computed is None:
        print("      Model event-based tidak punya baris yang cocok - berhenti.")
        return 1
    eb_rows, eb_risk, _eb_target = eb_computed
    eb_df = eb_rows[["installation_cycle_id"]].copy()
    eb_df["row_key"] = eb_rows.index.to_numpy()
    eb_df["risk_eb"] = eb_risk

    print("\n[3/4] Menggabungkan skor (irisan populasi kedua model)...")
    merged = static_df.merge(eb_df[["row_key", "risk_eb"]], on="row_key", how="inner")
    print(f"      Populasi irisan (kedua model punya skor): {len(merged):,} baris")

    target = merged["target"].to_numpy()
    risk_static = merged["risk_static"].to_numpy()
    risk_eb = merged["risk_eb"].to_numpy()

    import numpy as np
    import pandas as pd

    rank_static = pd.Series(risk_static).rank(pct=True).to_numpy()
    rank_eb = pd.Series(risk_eb).rank(pct=True).to_numpy()

    candidates = {
        "static_only (populasi irisan)": risk_static,
        "event_based_only (populasi irisan)": risk_eb,
        "ensemble_avg_raw": (risk_static + risk_eb) / 2,
        "ensemble_avg_rank": (rank_static + rank_eb) / 2,
        "ensemble_max": np.maximum(risk_static, risk_eb),
    }

    print("[4/4] Menghitung metrik operasional tiap kandidat (populasi SAMA, adil)...")
    from partrisk import config, training_utils

    rows_report = []
    for label, risk in candidates.items():
        m = training_utils.full_metrics(risk, risk, target, window_days, config.FAILURE_CAPACITY_PER_MONTH)
        rows_report.append({"label": label, **m})
        print(
            f"      {label:38s} PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}  "
            f"Recall@cap={m['recall_at_capacity']:.4f}  Precision@cap={m['precision_at_capacity']:.4f}"
        )

    report = ["# Ensemble operasional: model statis + event-based (Fase 4)", ""]
    report.append(
        f"Populasi irisan (kedua model punya skor pada baris yang sama): {len(merged):,} baris, "
        f"window {window_days:.0f} hari, kapasitas {config.FAILURE_CAPACITY_PER_MONTH}/bulan. "
        "`static_only`/`event_based_only` di sini dihitung ULANG pada populasi IRISAN (bukan angka lama "
        "dari evaluation_report.md masing-masing, yang populasinya beda) - supaya perbandingan adil."
    )
    report.append("")
    report.append(
        "| Kandidat | PR-AUC | ROC-AUC | Recall@cap | Precision@cap | Brier |\n"
        "|---|---|---|---|---|---|"
    )
    for r in rows_report:
        report.append(
            f"| {r['label']} | {r['pr_auc']:.4f} | {r['roc_auc']:.4f} | {r['recall_at_capacity']:.4f} | "
            f"{r['precision_at_capacity']:.4f} | {r['brier_calibrated']:.4f} |"
        )
    (EVENT_BASED_DIR / "reports" / "ensemble_operational.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\n[OK] Laporan: {EVENT_BASED_DIR / 'reports' / 'ensemble_operational.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
