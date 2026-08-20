"""Evaluasi model survival dalam dua lapis yang TIDAK dicampur (README Bagian 7-8).

    python survival_model/evaluate.py

Lapis 1 - native survival: C-index, Integrated Brier Score, Brier &
time-dependent AUC per horizon 30/60/90/120 hari, dihitung dari t=0=
installed_on pada VALIDATION dan TEST split lifecycle-level - cara standar
survival dievaluasi.

Lapis 2 - perbandingan ADIL dengan model classification production
(config.FAILURE_MODEL_DIR/CURRENT): meminjam populasi + label TEST
classification (`train.build_dataset()`, read-only, tidak pernah dipakai
fitting), menilai model survival di situ lewat risiko bersyarat
P(fail<=30d | survive sampai umur A) = 1-S(A+30)/S(A), lalu dievaluasi
dengan training_utils.full_metrics() yang SAMA PERSIS dipakai train.py
classification - supaya tidak membandingkan C-index vs ROC-AUC secara naif.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SURVIVAL_DIR = Path(__file__).resolve().parent
if str(SURVIVAL_DIR) not in sys.path:
    sys.path.insert(0, str(SURVIVAL_DIR))

import joblib
import numpy as np
import pandas as pd

import config
import training_utils

import build_dataset
from src import evaluation, features, utils

ARTIFACTS_DIR = SURVIVAL_DIR / "artifacts"
REPORTS_DIR = SURVIVAL_DIR / "reports"


def _load_root_module(name: str, filename: str):
    """Muat train.py DARI ROOT lewat file path, bukan `import train` biasa -
    survival_model/train.py sendiri punya nama file yang sama, dan sys.path
    berbasis nama akan ambigu. Ini menghindari itu sepenuhnya."""
    spec = importlib.util.spec_from_file_location(name, ROOT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_artifacts() -> dict:
    models = joblib.load(ARTIFACTS_DIR / "models.joblib")
    # Prediksi dipaksa single-thread: RandomSurvivalForest yang di-unpickle di
    # proses baru lalu diminta predict_survival_function() dengan n_jobs=-1
    # (bawaan saat training) terbukti membuat proses ini hang tanpa error saat
    # loky mencoba membongkar worker pool-nya - komputasinya sendiri selesai
    # dalam hitungan milidetik, hanya exit proses yang macet. Training TIDAK
    # kena masalah ini (tetap n_jobs=-1 di train.py) - hanya prediksi dari
    # model yang sudah disimpan/dimuat ulang.
    for model in models.values():
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
    encoder = joblib.load(ARTIFACTS_DIR / "encoder.joblib")
    y_train = joblib.load(ARTIFACTS_DIR / "y_train.joblib")
    metadata = json.loads((ARTIFACTS_DIR / "metadata.json").read_text(encoding="utf-8"))
    return {"models": models, "encoder": encoder, "y_train": y_train, "metadata": metadata}


# ---------------------------------------------------------------------------
# Lapis 1 - native survival
# ---------------------------------------------------------------------------


def native_layer(models: dict, encoder, y_train, dataset, feature_frame) -> dict:
    results: dict = {}
    for split_name in ("VALIDATION", "TEST"):
        mask = (dataset["split"] == split_name).to_numpy()
        from sksurv.util import Surv

        y_eval = Surv.from_arrays(
            event=dataset.loc[mask, "event_observed"].astype(bool).to_numpy(),
            time=dataset.loc[mask, "duration_days"].to_numpy(),
        )
        x_eval = features.encode(feature_frame.loc[mask], encoder)
        results[split_name] = {
            name: evaluation.native_metrics(model, y_train, x_eval, y_eval)
            for name, model in models.items()
        }
    return results


# ---------------------------------------------------------------------------
# Lapis 2 - perbandingan adil dengan classification model existing
# ---------------------------------------------------------------------------


def load_classification_test_rows() -> tuple:
    """Baris TEST classification (dipinjam READ-ONLY lewat `train.build_dataset()`
    di root, tidak pernah dipakai fitting) + panjang window uji dalam hari.

    Mahal (membangun ulang grid 30-harian 1,4 juta baris classification) -
    dipanggil SEKALI dan dipakai ulang oleh SEMUA model yang mau dinilai
    (evaluate.py maupun experiments.py yang menguji puluhan kandidat), bukan
    per model.
    """
    classification_train = _load_root_module("classification_train", "train.py")
    c_dataset, _c_features, _support, _data_end, _events, _cycles, _episodes = (
        classification_train.build_dataset()
    )
    test_rows = c_dataset.loc[c_dataset["split"] == classification_train.TEST].copy()
    observed = pd.to_datetime(test_rows["observation_on"])
    window_days = float((observed.max() - observed.min()).days) if len(test_rows) else 0.0
    return test_rows, window_days


def score_operational(
    model, feature_frame_by_cycle_id: pd.DataFrame, encoder, test_rows: pd.DataFrame, window_days: float,
    *, numeric_columns: list[str] | None = None,
) -> dict | None:
    """Skor SATU model survival pada populasi TEST classification yang
    dipinjam `load_classification_test_rows()` - `training_utils.full_metrics()`
    yang SAMA PERSIS dipakai `train.py` classification. Dipisah dari
    `classification_layer()` supaya `experiments.py` bisa memanggilnya
    berulang untuk banyak model kandidat tanpa membangun ulang `test_rows`.

    `feature_frame_by_cycle_id` = fitur baseline instalasi, index-nya
    `installation_cycle_id` (satu baris per lifecycle unik).
    """
    matched_mask = test_rows["installation_cycle_id"].isin(feature_frame_by_cycle_id.index)
    n_total, n_matched = len(test_rows), int(matched_mask.sum())
    if n_matched == 0:
        return None
    rows = test_rows.loc[matched_mask]
    ages = pd.to_numeric(rows["days_since_installation"], errors="coerce").to_numpy()
    target = rows["target_failure"].astype(bool).to_numpy()

    # Banyak baris TEST classification (grid 30-harian) berasal dari lifecycle
    # yang SAMA (satu PART diobservasi berkali-kali sepanjang siklusnya) -
    # kurva S(t) dihitung SEKALI per lifecycle unik (bukan per baris snapshot).
    # Tanpa dedup ini, predict_survival_function() pada puluhan ribu baris
    # sekaligus bisa mengalokasikan >1 GiB (uji coba pertama gagal karena ini).
    unique_ids = rows["installation_cycle_id"].drop_duplicates().to_numpy()
    unique_features = feature_frame_by_cycle_id.loc[unique_ids]
    x_unique = features.encode(unique_features, encoder, numeric_columns)

    times_grid, curves = utils.survival_curve_arrays(model, x_unique)
    curve_by_cycle = dict(zip(unique_ids, curves))
    risk_30d = np.array(
        [
            utils.conditional_risk(times_grid, curve_by_cycle[cid], age, 30.0)
            for cid, age in zip(rows["installation_cycle_id"].to_numpy(), ages)
        ]
    )
    metrics = training_utils.full_metrics(
        risk_30d, risk_30d, target, window_days, config.FAILURE_CAPACITY_PER_MONTH
    )
    metrics["rows_matched"] = n_matched
    metrics["rows_total_classification_test"] = n_total
    return metrics


def classification_layer(models: dict, encoder, built: dict) -> dict:
    print("      Membangun ulang dataset classification (dipinjam read-only, tidak di-fit)...")
    test_rows, window_days = load_classification_test_rows()

    survival_dataset = built["dataset"]
    survival_features = built["features"].copy()
    survival_features.index = survival_dataset["installation_cycle_id"].to_numpy()

    n_matched = int(test_rows["installation_cycle_id"].isin(survival_features.index).sum())
    print(
        f"      {n_matched:,}/{len(test_rows):,} baris TEST classification punya lifecycle survival yang "
        "cocok (sisanya di-exclude survival tapi tidak di classification, mis. karena aturan "
        "censoring per-split - lihat reports/data_validation.md)"
    )

    results = {
        name: score_operational(model, survival_features, encoder, test_rows, window_days)
        for name, model in models.items()
    }
    any_result = next((m for m in results.values() if m is not None), None)

    production_version, production_metrics = _load_classification_production_metrics()
    return {
        "rows_matched": any_result["rows_matched"] if any_result else 0,
        "rows_total_classification_test": len(test_rows),
        "window_days": window_days,
        "survival_models": results,
        "classification_production_version": production_version,
        "classification_production_metrics": production_metrics,
    }


def _load_classification_production_metrics() -> tuple[str | None, dict | None]:
    pointer = config.FAILURE_MODEL_DIR / "CURRENT"
    if not pointer.exists():
        return None, None
    version = pointer.read_text(encoding="utf-8").strip()
    metadata_path = config.FAILURE_MODEL_DIR / version / "metadata.json"
    if not metadata_path.exists():
        return version, None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    candidate = metadata.get("promotion_comparison", {}).get("candidate")
    if candidate is None:
        candidate = {**metadata["evaluation_metrics"]["test"]}
    return version, candidate


# ---------------------------------------------------------------------------


def _format_report(native: dict, comparison: dict, metadata: dict) -> str:
    lines = ["# Laporan evaluasi survival model", ""]

    lines.append("## Lapis 1 - native survival (dari t=0=installed_on)")
    for split_name, per_model in native.items():
        lines.append(f"\n### {split_name}")
        for model_name, m in per_model.items():
            uno = f"{m['uno_c_index']:.4f}" if m.get("uno_c_index") is not None else "N/A"
            lines.append(
                f"- **{model_name}**: rows={m['rows']:,} events={m['events']:,} "
                f"C-index(Harrell)={m['c_index']:.4f} C-index(Uno/IPCW)={uno} "
                f"IBS={m['integrated_brier_score']}"
            )
            if m["brier_at_horizon"]:
                brier = ", ".join(f"{h}d={v:.4f}" for h, v in m["brier_at_horizon"].items())
                lines.append(f"  - Brier per horizon: {brier}")
            if m["time_dependent_auc_at_horizon"]:
                auc = ", ".join(f"{h}d={v:.4f}" for h, v in m["time_dependent_auc_at_horizon"].items())
                lines.append(f"  - Time-dependent AUC per horizon: {auc}")
            if not m["horizons_evaluable_days"]:
                lines.append(
                    "  - (horizon 30/60/90/120 hari tidak dapat dihitung - follow-up split ini "
                    "lebih pendek dari horizon tsb.)"
                )

    lines.append("\n## Lapis 2 - perbandingan adil vs classification model (populasi TEST classification)")
    lines.append(
        f"\n{comparison['rows_matched']:,} dari {comparison['rows_total_classification_test']:,} baris "
        f"TEST classification cocok dengan lifecycle survival ({comparison['window_days']:.0f} hari window, "
        f"kapasitas {config.FAILURE_CAPACITY_PER_MONTH}/bulan)."
    )
    header = f"{'model':28s} {'PR-AUC':>8s} {'ROC-AUC':>8s} {'Recall@cap':>11s} {'Precision@cap':>14s} {'Brier':>8s}"
    lines.append(f"\n```\n{header}")
    for name, m in comparison["survival_models"].items():
        if m is None:
            lines.append(f"{name:28s} (tidak ada baris TEST classification yang cocok)")
            continue
        lines.append(
            f"{name:28s} {m['pr_auc']:>8.4f} {m['roc_auc']:>8.4f} {m['recall_at_capacity']:>11.4f} "
            f"{m['precision_at_capacity']:>14.4f} {m['brier_calibrated']:>8.4f}"
        )
    prod = comparison["classification_production_metrics"]
    if prod:
        label = f"classification ({comparison['classification_production_version']})"
        lines.append(
            f"{label:28s} {prod.get('pr_auc', float('nan')):>8.4f} {prod.get('roc_auc', float('nan')):>8.4f} "
            f"{prod.get('recall_at_capacity', float('nan')):>11.4f} "
            f"{prod.get('precision_at_capacity', float('nan')):>14.4f} "
            f"{prod.get('brier_calibrated', float('nan')):>8.4f}"
        )
    lines.append("```")
    lines.append(
        "\nCatatan: skor survival di sini pakai fitur baseline INSTALASI (bukan fitur yang "
        "di-refresh ke tanggal snapshot seperti classification) - lihat README bagian "
        "\"Keterbatasan: baseline instalasi vs kondisi sekarang\". Perbandingan adil dari sisi "
        "horizon/populasi/label, tapi classification model punya keuntungan struktural (fitur "
        "lebih segar)."
    )
    return "\n".join(lines)


def main() -> int:
    print("[1/3] Memuat artifacts hasil train.py...")
    artifacts = load_artifacts()

    print("[2/3] Lapis 1: metrik native survival (VALIDATION & TEST)...")
    built = build_dataset.build()
    native = native_layer(
        artifacts["models"], artifacts["encoder"], artifacts["y_train"], built["dataset"], built["features"]
    )
    for split_name, per_model in native.items():
        for model_name, m in per_model.items():
            print(f"      {split_name:10s} {model_name:24s} C-index={m['c_index']:.4f}")

    print("[3/3] Lapis 2: perbandingan adil dengan classification model production...")
    comparison = classification_layer(artifacts["models"], artifacts["encoder"], built)

    report = _format_report(native, comparison, artifacts["metadata"])
    print("\n" + report)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "evaluation_report.md").write_text(report, encoding="utf-8")
    print(f"\n[OK] Laporan tersimpan di {REPORTS_DIR / 'evaluation_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
