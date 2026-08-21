"""Fase A3 (plan restrukturisasi): studi kalibrasi untuk model compact event-based
(konfigurasi pemenang A2 - lihat `experiments_compact_model.py`, 66,2 MB,
VAL C-index 0,8417).

`1 - S(h)` dari RSF adalah frekuensi relatif (rata-rata cumulative hazard antar
pohon) - BELUM tentu cocok dengan probabilitas empiris pada populasi VALIDATION
(sama seperti CatBoost raw score juga dikalibrasi lewat IsotonicRegression di
`train.py:156-157` sebelum dipakai `risk_cutoffs`/dashboard). Skrip ini
mengukur SEBERAPA JAUH kalibrasi mengubah keputusan operasional, BUKAN
menghasilkan artifact produksi (itu Fase C, kalau restrukturisasi lanjut).

Empat isotonic regressor terpisah (satu per horizon 30/60/90/120 hari), TIAP
horizon dikalibrasi independen dari label biner definitif:

    label(h) = 1  kalau event_observed & duration_days <= h   (gagal sebelum h)
    label(h) = 0  kalau duration_days >= h                    (masih hidup di h,
                                                                 event ATAU censored,
                                                                 sama-sama valid)
    label(h) = dibuang kalau censored SEBELUM h (duration_days < h, event_observed=0)
                                                                 - tidak diketahui apa
                                                                 yang terjadi antara
                                                                 censor dan h, BUKAN
                                                                 dipaksa jadi 0/1.

Karena tiap horizon dikalibrasi TERPISAH, kurva kalibrasi antar horizon BISA
saling silang (mis. calibrated_60d < calibrated_30d di satu baris) walau
S(t) mentahnya monoton turun - makanya `cummax` lintas horizon WAJIB
(lihat `tests/test_parity.py:132-139`, aturan yang sama berlaku di sini).

    python survival_model/event_based/experiments_calibration_study.py
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

import os

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

import build_dataset
import config
from src import model_fit
from src import utils as survival_utils

from eb_src import features
from experiments_compact_model import COMPACT_RSF_PARAMS, coarsen_duration_days

REPORTS_DIR = EVENT_BASED_DIR / "reports"
HORIZONS = [30, 60, 90, 120]


def _label_at_horizon(duration_days: np.ndarray, event_observed: np.ndarray, horizon: float):
    """Lihat docstring modul untuk definisi label. Baris censored sebelum
    horizon dikembalikan sebagai NaN (dibuang oleh pemanggil), bukan 0/1."""
    label = np.full(len(duration_days), np.nan)
    label[event_observed & (duration_days <= horizon)] = 1.0
    label[duration_days >= horizon] = 0.0
    return label


def main() -> int:
    print("[1/4] Menyusun dataset (cache lokal untuk kecepatan studi - lihat A2 soal G8 untuk artifact final)...")
    os.environ.setdefault("SURVIVAL_BUILD_CACHE", "1")
    built = build_dataset.build()
    dataset, feature_frame = built["dataset"], built["features"]

    train_mask = (dataset["split"] == "TRAIN").to_numpy()
    val_mask = (dataset["split"] == "VALIDATION").to_numpy()

    print("[2/4] Melatih kandidat compact (konfigurasi pemenang A2)...")
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv

    encoder = features.fit_encoder(feature_frame.loc[train_mask])
    x_train = features.encode(feature_frame.loc[train_mask], encoder)
    x_val = features.encode(feature_frame.loc[val_mask], encoder)

    y_train_true = model_fit.make_survival_target(dataset, train_mask)
    y_train_coarse = Surv.from_arrays(
        event=y_train_true["event"], time=coarsen_duration_days(y_train_true["time"])
    )
    model = RandomSurvivalForest(**COMPACT_RSF_PARAMS).fit(x_train, y_train_coarse)

    print("[3/4] Menghitung S(h) VALIDATION pada 30/60/90/120 hari + kalibrasi per horizon...")
    times_grid, curves = survival_utils.survival_curve_arrays(model, x_val)
    surv_at_horizons = survival_utils.step_eval_matrix(times_grid, curves, HORIZONS)
    raw_risk = 1.0 - surv_at_horizons  # [n_val, 4]

    val_duration = dataset.loc[val_mask, "duration_days"].to_numpy()
    val_event = dataset.loc[val_mask, "event_observed"].to_numpy().astype(bool)

    calibrated = np.full_like(raw_risk, np.nan)
    rows = []
    for j, h in enumerate(HORIZONS):
        label = _label_at_horizon(val_duration, val_event, float(h))
        usable = ~np.isnan(label)
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_risk[usable, j], label[usable])
        calibrated[:, j] = calibrator.predict(raw_risk[:, j])

        raw_brier = brier_score_loss(label[usable], raw_risk[usable, j])
        cal_brier = brier_score_loss(label[usable], calibrated[usable, j])
        rows.append((h, int(usable.sum()), int(label[usable].sum()), raw_brier, cal_brier))

    print("[4/4] cummax lintas horizon + dampak ke risk_cutoffs 0,25/0,15...")
    violations_before_cummax = int((np.diff(calibrated, axis=1) < -1e-12).any(axis=1).sum())
    calibrated_monotone = np.maximum.accumulate(calibrated, axis=1)
    still_violating = int((np.diff(calibrated_monotone, axis=1) < -1e-12).any(axis=1).sum())

    high = config.FAILURE_HIGH_PROBABILITY_THRESHOLD
    medium = config.FAILURE_MEDIUM_PROBABILITY_THRESHOLD
    raw_30 = raw_risk[:, 0]
    cal_30 = calibrated_monotone[:, 0]
    raw_high, raw_medium = int((raw_30 >= high).sum()), int(((raw_30 >= medium) & (raw_30 < high)).sum())
    cal_high, cal_medium = int((cal_30 >= high).sum()), int(((cal_30 >= medium) & (cal_30 < high)).sum())

    cal_lines = ["| Horizon | Baris terpakai | Kejadian | Brier mentah | Brier terkalibrasi |", "|---|---|---|---|---|"]
    for h, n, n_event, raw_b, cal_b in rows:
        cal_lines.append(f"| {h}d | {n:,} | {n_event:,} | {raw_b:.4f} | {cal_b:.4f} |")
    cal_table = "\n".join(cal_lines)

    print()
    print(cal_table)
    print()
    print(f"      Pelanggaran monotonisitas SEBELUM cummax: {violations_before_cummax}/{len(calibrated):,} baris")
    print(f"      Pelanggaran SESUDAH cummax: {still_violating} (harus 0)")
    print()
    print(f"      risk_cutoffs 30d pada populasi VALIDATION ({len(raw_30):,} baris landmark):")
    print(f"        mentah        : HIGH={raw_high}  MEDIUM={raw_medium}")
    print(f"        terkalibrasi  : HIGH={cal_high}  MEDIUM={cal_medium}")

    assert still_violating == 0, "cummax gagal menjamin monotonisitas - periksa NaN di calibrated"

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = f"""# Fase A3: studi kalibrasi (event-based, konfigurasi compact A2)

Populasi: VALIDATION landmark rows (aturan proyek - keputusan tidak pernah dari TEST).
Model: kandidat compact A2 (n_estimators={COMPACT_RSF_PARAMS['n_estimators']},
min_samples_leaf={COMPACT_RSF_PARAMS['min_samples_leaf']}, grid dikasarkan).

## Brier per horizon, mentah vs terkalibrasi (isotonic independen per horizon)

{cal_table}

## Monotonisitas lintas horizon (30<=60<=90<=120)

- Pelanggaran SEBELUM cummax: {violations_before_cummax}/{len(calibrated):,} baris
  (isotonic per horizon dikalibrasi TERPISAH, jadi ini diharapkan bukan 0 - lihat
  docstring skrip). cummax lintas [30,60,90,120] WAJIB, sama seperti
  `tests/test_parity.py:132-139` menegaskan untuk CatBoost hazard-chaining.
- Pelanggaran SESUDAH cummax: {still_violating} (harus 0 - diverifikasi lewat assert).

## Dampak ke risk_cutoffs (HIGH>={high}, MEDIUM>={medium}) pada risiko 30 hari

| | HIGH | MEDIUM |
|---|---|---|
| Skor mentah (1-S(30)) | {raw_high} | {raw_medium} |
| Skor terkalibrasi | {cal_high} | {cal_medium} |

Populasi VALIDATION landmark ({len(raw_30):,} baris) - BUKAN populasi PART aktif
production (itu perlu skor pada `predict.py`-style observation_on, lihat A1),
cuma untuk melihat ARAH dan BESAR pergeseran akibat kalibrasi.
"""
    (REPORTS_DIR / "gate_a3_calibration_study.md").write_text(report, encoding="utf-8")
    print(f"\n[OK] Laporan: {REPORTS_DIR / 'gate_a3_calibration_study.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
