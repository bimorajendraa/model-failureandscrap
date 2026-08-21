"""Perbandingan ADIL model survival dengan model classification production
(config.FAILURE_MODEL_DIR/CURRENT): meminjam populasi + label TEST
classification (`training.failure_classification.build_dataset()`,
read-only, TIDAK PERNAH dipakai fitting), menilai model survival di situ
lewat risiko bersyarat P(fail<=30d | survive sampai umur A) = 1-S(A+30)/S(A),
lalu dievaluasi dengan `training.versioning.full_metrics()` yang SAMA PERSIS
dipakai training classification - supaya tidak membandingkan C-index vs
ROC-AUC secara naif.

Diekstrak dari `survival_model/evaluate.py` (model statis, sudah dihapus -
event-based menang di semua metrik operasional, lihat gate_decision.md) ke
sini (Fase C1 restrukturisasi) karena ini "mesin pembanding adil" yang
dipakai berulang, bukan kode eksperimen sekali pakai - `score_operational()`
dipanggil untuk SETIAP model survival kandidat yang mau dinilai pada
window/populasi yang identik.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from partrisk import config
from partrisk.features.survival import builder as features
from partrisk.survival import curves
from partrisk.training import failure_classification as classification_train
from partrisk.training import versioning as training_utils


def load_classification_test_rows() -> tuple:
    """Baris TEST classification (dipinjam READ-ONLY lewat
    `training.failure_classification.build_dataset()`, tidak pernah dipakai
    fitting) + panjang window uji dalam hari.

    Mahal (membangun ulang grid 30-harian 1,4 juta baris classification) -
    dipanggil SEKALI dan dipakai ulang oleh SEMUA model survival yang mau
    dinilai, bukan per model.
    """
    c_dataset, _c_features, _support, _data_end, _events, _cycles, _episodes = (
        classification_train.build_dataset()
    )
    test_rows = c_dataset.loc[c_dataset["split"] == classification_train.TEST].copy()
    observed = pd.to_datetime(test_rows["observation_on"])
    window_days = float((observed.max() - observed.min()).days) if len(test_rows) else 0.0
    return test_rows, window_days


def compute_risk_30d(
    model, feature_frame_by_cycle_id: pd.DataFrame, encoder, test_rows: pd.DataFrame,
    *, numeric_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray] | None:
    """Skor MENTAH (risk_30d per baris, BELUM diringkas jadi metrik) SATU
    model survival pada populasi TEST classification. Diekstrak dari
    `score_operational()` (yang tetap dipertahankan APA ADANYA untuk
    caller lama) supaya skor mentah ANTAR MODEL bisa digabung (ensemble)
    SEBELUM dihitung metriknya.

    Return `(rows, risk_30d, target)` - `rows` (subset `test_rows` yang
    matched) dipakai caller untuk join ANTAR model lewat
    `installation_cycle_id` (urutan baris BEDA-BEDA antar model kalau
    populasi matched-nya beda, TIDAK boleh digabung by POSITION)."""
    matched_mask = test_rows["installation_cycle_id"].isin(feature_frame_by_cycle_id.index)
    if int(matched_mask.sum()) == 0:
        return None
    rows = test_rows.loc[matched_mask]
    ages = pd.to_numeric(rows["days_since_installation"], errors="coerce").to_numpy()
    target = rows["target_failure"].astype(bool).to_numpy()

    # Banyak baris TEST classification (grid 30-harian) berasal dari lifecycle
    # yang SAMA (satu PART diobservasi berkali-kali sepanjang siklusnya) -
    # kurva S(t) dihitung SEKALI per lifecycle unik (bukan per baris snapshot).
    # Tanpa dedup ini, predict_survival_function() pada puluhan ribu baris
    # sekaligus bisa mengalokasikan >1 GiB.
    unique_ids = rows["installation_cycle_id"].drop_duplicates().to_numpy()
    unique_features = feature_frame_by_cycle_id.loc[unique_ids]
    x_unique = features.encode(unique_features, encoder, numeric_columns)

    times_grid, curve_values = curves.survival_curve_arrays(model, x_unique)
    curve_by_cycle = dict(zip(unique_ids, curve_values))
    risk_30d = np.array(
        [
            curves.conditional_risk(times_grid, curve_by_cycle[cid], age, 30.0)
            for cid, age in zip(rows["installation_cycle_id"].to_numpy(), ages)
        ]
    )
    return rows, risk_30d, target


def score_operational(
    model, feature_frame_by_cycle_id: pd.DataFrame, encoder, test_rows: pd.DataFrame, window_days: float,
    *, numeric_columns: list[str] | None = None,
) -> dict | None:
    """Skor SATU model survival pada populasi TEST classification yang
    dipinjam `load_classification_test_rows()` - `training.versioning.full_metrics()`
    yang SAMA PERSIS dipakai training classification. Dipisah dari
    `compute_risk_30d()` supaya pemanggil bisa mengulang untuk banyak model
    kandidat tanpa membangun ulang `test_rows`.

    `feature_frame_by_cycle_id` = fitur baseline instalasi, index-nya
    `installation_cycle_id` (satu baris per lifecycle unik).
    """
    computed = compute_risk_30d(model, feature_frame_by_cycle_id, encoder, test_rows, numeric_columns=numeric_columns)
    if computed is None:
        return None
    rows, risk_30d, target = computed
    metrics = training_utils.full_metrics(
        risk_30d, risk_30d, target, window_days, config.FAILURE_CAPACITY_PER_MONTH
    )
    metrics["rows_matched"] = len(rows)
    metrics["rows_total_classification_test"] = len(test_rows)
    return metrics
