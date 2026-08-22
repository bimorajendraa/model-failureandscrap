"""Latih model survival event-based (RSF + Cox PH) pada landmark PART.

    python -m partrisk.training.failure_survival

Reuse TOTAL logic fitting/evaluasi native dari `survival.model_fit` (TIDAK
diubah, TIDAK disalin) - satu-satunya yang beda dari model classification
adalah sumber datanya (`training.datasets.survival`, banyak baris/lifecycle
per lifecycle) dan encoder/fitur (`features.survival.builder`, kolom
categorical/numeric berbeda karena umur pemasangan sekarang jadi fitur,
bukan sumbu waktu konstan).

Konfigurasi RSF di sini adalah kandidat COMPACT pemenang Fase A2 (lihat
survival_model/event_based/reports/gate_a2_compact_model.md /
gate_decision.md), BUKAN default riset penuh di `survival.model_fit`:

- Target `duration_days` yang dilihat RSF.fit() DIKASARKAN (resolusi harian
  s/d 120 hari - horizon kontrak API, kelipatan 60 hari di atasnya). Evaluasi
  (C-index/IBS/Brier/AUC) TETAP pakai duration_days ASLI - HANYA yang dilihat
  .fit() yang dikasarkan. Diverifikasi A2: artifact 5,26 GB -> 66,2 MB (79x)
  dengan C-index VALIDATION yang JUSTRU lebih baik (grid lebih kasar
  bertindak sebagai regularisasi, bukan kompromi akurasi-vs-ukuran).
- n_estimators=50, min_samples_leaf=100 (vs default 100/30) - lewat
  parameter `params` bawaan `model_fit.fit_models()`, BUKAN mengubah
  `DEFAULT_RSF_PARAMS` (itu tetap dipakai skrip riset lama/model lain yang
  belum eksplisit override).

Kalibrasi 4 isotonic regressor (horizon 30/60/90/120, populasi VALIDATION,
label biner definitif - lihat `_label_at_horizon`) + cummax lintas horizon
disimpan sebagai `calibrators.joblib` (Fase A3, lihat gate_a3_calibration_study.md)
- BELUM dipakai `predict/survival.py` (field advisory sengaja TIDAK
  dikalibrasi dulu, curve_is_calibrated=False di kontrak API kalau/ketika
  field itu ditambahkan) - disimpan untuk dipakai nanti tanpa perlu
  retrain ulang.

Mode ADITIF (lihat gate_decision.md) - model ini TIDAK menggantikan CatBoost.
ARTIFACTS_DIR masih menunjuk artifact riset (survival_model/event_based/artifacts/),
BUKAN models/failure/v3/ - tidak ada mekanisme cutover/rollback yang perlu
dibangun untuk field advisory saja.

Kebijakan retrain (Fase R3 upgrade RSF, reports/rsf_r1_evaluation.md dkk):
model ini ADVISORY - tidak mengatur ranking/urutan inspeksi (itu tetap CatBoost),
jadi TIDAK perlu retrain tiap minggu seperti CatBoost. Retrain wajar: bulanan,
atau kapan pun `data_end` (lihat metadata.json) sudah bergeser jauh (mis. >60
hari) dari training terakhir. Retrain LEBIH SERING tidak salah, hanya tidak perlu.

Gate promosi (R3, ringan - BUKAN dual-gate PR-AUC/Recall seperti CatBoost
`decide_promotion`, karena model ini tidak dipakai untuk ranking): artifact
kandidat HANYA menggantikan artifact production kalau Brier@30d DAN Brier@90d
TEST tidak memburuk dibanding artifact yang sudah ada (`decide_survival_promotion()`
di bawah). Kalau gagal, artifact LAMA tetap dipakai - training TIDAK
menimpa file secara diam-diam.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sksurv.util import Surv

from partrisk import config
from partrisk.features.survival import builder as features
from partrisk.survival import curves, model_fit
from partrisk.training.datasets import survival as build_dataset

ARTIFACTS_DIR = config.PACKAGE_DIR / "survival_model" / "event_based" / "artifacts"

CALIBRATION_HORIZONS_DAYS = [30, 60, 90, 120]

# Pemenang Fase A2 (grid dikasarkan + forest lebih kecil) - lihat docstring
# modul. n_jobs=1 (bukan -1): artifact hasil training langsung dipakai
# predict/survival.py di proses yang sama tanpa unpickle ulang, tapi
# disamakan dengan default aman evaluate.py/predict.py (RSF ter-unpickle +
# n_jobs=-1 + predict_survival_function() hang tanpa error).
COMPACT_RSF_PARAMS = dict(
    n_estimators=50,
    min_samples_split=140,
    min_samples_leaf=100,
    max_features="sqrt",
    n_jobs=1,
    random_state=42,
)


def coarsen_duration_days(days: np.ndarray) -> np.ndarray:
    """Resolusi harian s/d 120 hari (horizon kontrak API), kelipatan 60 hari
    di atasnya. HANYA dipakai untuk target yang dilihat RSF.fit() - evaluasi
    tetap pakai duration_days asli."""
    days = np.asarray(days, dtype=float)
    near = np.round(days)
    far = 120.0 + np.round((days - 120.0) / 60.0) * 60.0
    return np.maximum(np.where(days <= 120.0, near, far), 1.0)


def _label_at_horizon(duration_days: np.ndarray, event_observed: np.ndarray, horizon: float) -> np.ndarray:
    """Label biner definitif untuk kalibrasi - lihat gate_a3_calibration_study.md.

    1  kalau event_observed & duration_days <= horizon  (gagal sebelum horizon)
    0  kalau duration_days >= horizon                    (masih hidup di horizon,
                                                            event ATAU censored)
    NaN kalau censored SEBELUM horizon - dibuang oleh pemanggil (tidak
    diketahui apa yang terjadi antara censor dan horizon), BUKAN dipaksa 0/1.
    """
    label = np.full(len(duration_days), np.nan)
    label[event_observed & (duration_days <= horizon)] = 1.0
    label[duration_days >= horizon] = 0.0
    return label


def fit_calibrators(model, x_val, val_duration: np.ndarray, val_event: np.ndarray) -> dict[int, IsotonicRegression]:
    """Satu IsotonicRegression per horizon, populasi VALIDATION. Cummax
    lintas horizon adalah tanggung jawab PEMANGGIL saat memakai kalibrator
    ini (isotonic per horizon dikalibrasi TERPISAH, kurvanya BISA saling
    silang walau S(t) mentahnya monoton turun - lihat gate_a3_calibration_study.md)."""
    times_grid, curve_values = curves.survival_curve_arrays(model, x_val)
    surv_at_horizons = curves.step_eval_matrix(times_grid, curve_values, CALIBRATION_HORIZONS_DAYS)
    raw_risk = 1.0 - surv_at_horizons

    calibrators: dict[int, IsotonicRegression] = {}
    for j, h in enumerate(CALIBRATION_HORIZONS_DAYS):
        label = _label_at_horizon(val_duration, val_event, float(h))
        usable = ~np.isnan(label)
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_risk[usable, j], label[usable])
        calibrators[h] = calibrator
    return calibrators


def decide_survival_promotion(candidate_test_metrics: dict, incumbent_test_metrics: dict | None) -> tuple[bool, str]:
    """Gate RINGAN Fase R3 - model ini advisory (tidak mengatur ranking), jadi
    BUKAN dual-gate PR-AUC/Recall ala `training.versioning.decide_promotion`.
    Kandidat menang HANYA kalau Brier@30d DAN Brier@90d TEST tidak memburuk
    dibanding artifact production sekarang - dua horizon itu yang paling
    relevan untuk field advisory (risk_30d/90d, median/90pct sisa umur).

    `candidate_test_metrics`/`incumbent_test_metrics` = `metrics["random_survival_forest"]["test"]`
    dari `model_fit.evaluate_models()` (native_metrics - kunci brier_at_horizon
    dict[int,float]) atau, untuk incumbent lama, `metadata.json` hasil json.load
    (kunci brier_at_horizon dict[str,float] - dinormalkan ke int di bawah)."""
    if incumbent_test_metrics is None:
        return True, "belum ada artifact production sebelumnya"

    def brier(metrics: dict, horizon: int) -> float:
        table = metrics["brier_at_horizon"]
        return float(table.get(horizon, table.get(str(horizon))))

    b30_candidate, b30_incumbent = brier(candidate_test_metrics, 30), brier(incumbent_test_metrics, 30)
    b90_candidate, b90_incumbent = brier(candidate_test_metrics, 90), brier(incumbent_test_metrics, 90)
    reason = (
        f"Brier@30d {b30_candidate:.4f} vs incumbent {b30_incumbent:.4f} | "
        f"Brier@90d {b90_candidate:.4f} vs incumbent {b90_incumbent:.4f}"
    )
    if b30_candidate <= b30_incumbent and b90_candidate <= b90_incumbent:
        return True, reason
    return False, reason


def main() -> int:
    print("[1/5] Menyusun dataset event-based (baca database, landmark)...")
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

    print("[2/5] Encoding fitur (one-hot kategorikal, fit di TRAIN saja)...")
    encoder = features.fit_encoder(feature_frame.loc[train_mask])
    x_train = features.encode(feature_frame.loc[train_mask], encoder)
    x_val = features.encode(feature_frame.loc[val_mask], encoder)
    x_test = features.encode(feature_frame.loc[test_mask], encoder)
    y_train = model_fit.make_survival_target(dataset, train_mask)
    y_val = model_fit.make_survival_target(dataset, val_mask)
    y_test = model_fit.make_survival_target(dataset, test_mask)

    print("[3/5] Melatih RSF (kandidat compact A2) + Cox PH (landmark, banyak baris/lifecycle)...")
    y_train_coarse = Surv.from_arrays(event=y_train["event"], time=coarsen_duration_days(y_train["time"]))
    models = model_fit.fit_models(
        x_train, y_train_coarse, params={"random_survival_forest": COMPACT_RSF_PARAMS}
    )
    # evaluate_models() butuh y_train UNTUK SEMUA model dari satu array yang
    # sama (dipakai model_fit.evaluate_models untuk estimasi IPCW/Uno-C) -
    # RSF sudah di-fit dengan target dikasarkan di atas, tapi EVALUASI
    # (baris ini) tetap pakai y_train ASLI (tidak dikasarkan) supaya angka
    # C-index/IBS/Brier/AUC jujur dan sebanding dengan konfigurasi riset lama.
    metrics = model_fit.evaluate_models(models, y_train, x_val, y_val, x_test, y_test)
    for name in models:
        print(
            f"      {name:24s} C-index(full landmark) val={metrics[name]['validation']['c_index']:.4f}  "
            f"test={metrics[name]['test']['c_index']:.4f}"
        )

    print("[4/5] Kalibrasi RSF (4 isotonic, populasi VALIDATION)...")
    val_duration = dataset.loc[val_mask, "duration_days"].to_numpy()
    val_event = dataset.loc[val_mask, "event_observed"].to_numpy().astype(bool)
    calibrators = fit_calibrators(models["random_survival_forest"], x_val, val_duration, val_event)
    for h, calibrator in calibrators.items():
        print(f"      horizon={h}d - {len(calibrator.X_thresholds_)} titik kalibrasi")

    print("[Gate R3] Membandingkan kandidat dengan artifact production (kalau ada)...")
    incumbent_metadata_path = ARTIFACTS_DIR / "metadata.json"
    incumbent_test_metrics = None
    if incumbent_metadata_path.exists():
        incumbent_metadata = json.loads(incumbent_metadata_path.read_text(encoding="utf-8"))
        incumbent_test_metrics = incumbent_metadata["evaluation_metrics_full_landmark_rows"]["random_survival_forest"]["test"]
    approved, gate_reason = decide_survival_promotion(metrics["random_survival_forest"]["test"], incumbent_test_metrics)
    print(f"      {gate_reason}")
    if not approved:
        print("      DITOLAK - Brier@30d/90d TEST memburuk dibanding artifact production sekarang.")
        print("      Artifact TIDAK ditimpa - artifact lama tetap dipakai serving.")
        return 1
    print("      DITERIMA - artifact production akan ditimpa.")

    print("[5/5] Menyimpan artifacts...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, ARTIFACTS_DIR / "models.joblib")
    joblib.dump(encoder, ARTIFACTS_DIR / "encoder.joblib")
    joblib.dump(y_train, ARTIFACTS_DIR / "y_train.joblib")
    joblib.dump(calibrators, ARTIFACTS_DIR / "calibrators.joblib")

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
            "random_survival_forest": COMPACT_RSF_PARAMS,
            "random_survival_forest_target_coarsening": (
                "daily resolution <=120 days, 60-day steps beyond - fit() target only, "
                "NOT applied to evaluation (native_metrics uses original duration_days)"
            ),
            "cox_ph": model_fit.DEFAULT_COX_PARAMS,
        },
        "calibration": {
            "method": "isotonic per horizon, VALIDATION landmark rows, definite binary label "
            "(event<=horizon=1, survived-to-horizon=0, censored-before-horizon excluded)",
            "horizons_days": CALIBRATION_HORIZONS_DAYS,
            "cummax_required": True,
            # Fase R1 upgrade RSF (commit be83a03): predict/survival.py
            # SEKARANG memakai calibrators.joblib untuk calibrated_risk_*d -
            # lihat _calibrate_risk(). Dulu False (dilatih tapi belum dipakai).
            "applied_to_advisory_fields": True,
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
