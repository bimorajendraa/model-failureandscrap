"""Fitur event-based: SAMA seperti fitur final model statis
(`survival_model/src/features.py`) DITAMBAH umur pemasangan
(`log_days_since_installation`/`installation_age_band`).

Model statis men-drop 2 fitur itu karena SELALU konstan (umur=0, sebab
observation_on==installed_on selalu di sana - lihat
survival_model/src/features.py). Di sini observation_on = landmark
(`eb_src.landmark_builder`), BUKAN installed_on lagi - umur pemasangan jadi
sinyal UTAMA (persis Tahap 8: "current age" adalah fitur kondisi PART
pertama yang diminta), jadi DIPERTAHANKAN, bukan di-drop.

Reuse SEPENUHNYA (tidak ada logic baru untuk hal yang sudah ada):
- `feature_builder.attach_history`/`attach_fleet`: sudah generik terhadap
  kolom `observation_on` - dipanggil dengan observation_on=landmark, bukan
  installed_on, otomatis menghitung ulang riwayat/armada PADA UMUR itu.
- `install_context.attach_install_context` (survival_model/src, parent):
  konteks instalasi KONSTAN per lifecycle - benar untuk di-merge apa adanya
  ke semua landmark milik lifecycle yang sama.
- `previous_cycle` (survival_model/src, parent): previous-cycle KONSTAN per
  lifecycle (bicara tentang siklus SEBELUMNYA, bukan siklus berjalan) -
  benar untuk di-merge apa adanya ke semua landmark.
- Threshold kategori (part_model=200, item_type=300): dipakai APA ADANYA
  dari `survival_model/src/features.FINAL_CATEGORY_THRESHOLDS` - hasil
  sweep VALIDATION yang sudah divalidasi utuh untuk populasi lifecycle
  survival. TIDAK di-sweep ulang khusus populasi landmark (proporsional -
  populasi dasarnya sama, hanya jumlah baris per lifecycle yang berbeda;
  re-sweep adalah kandidat penyempurnaan lanjutan, bukan blocker).

TIDAK reuse (butuh logic baru, didokumentasikan di bawah):
- Dukungan historis (`part_model_category`/`item_type_at_install_grouped`)
  TIDAK boleh dihitung lewat `feature_builder.cumulative_support`/
  `categorical_support.cumulative_support` langsung pada frame landmark -
  keduanya me-rank tiap BARIS dalam frame yang diberikan, dan satu lifecycle
  di sini menghasilkan banyak baris (landmark) - kalau dipakai apa adanya,
  "dukungan" akan menghitung landmark yang sama berkali-kali seolah banyak
  instalasi baru terjadi, bukan satu. `point_in_time_support()` di bawah
  menghitung dukungan yang BENAR: jumlah LIFECYCLE (bukan baris landmark)
  dengan installed_on <= observation_on landmark ini.

Fitur DINAMIS TAMBAHAN (hasil `experiments.py` ablation - konfigurasi
"G_combined_without_device", VAL t0-only RSF 0,7849 -> 0,7954, lihat
reports/dynamic_ablation.md dan reports/g_without_device.md): degradation
trend + cumulative physical usage + jendela corrective 60/90 hari, DIHITUNG
DI `eb_src/dynamic_history.py`, ditempel di sini lewat `attach_dynamic_extra()`.

Fitur DEVICE/TERMINAL (`terminal_type_grouped`, konfigurasi "F_combined_all",
VAL t0-only 0,8036) - AWALNYA diambil dari schema `analytics` (riset lama),
`config.py` eksplisit melarang production bergantung ke schema itu. SEKARANG
sudah direproduksi APA ADANYA sebagai query kanonikal
`data_reader.get_terminal_context()` (dibangun ulang dari tabel mentah
`journal.t_item_request_out`/`master.t_mtr_item`/`inventory.t_item`,
diverifikasi angkanya PERSIS sama dengan schema `analytics` - lihat
docstring `data_reader.get_terminal_context()`) - production TIDAK lagi
bergantung ke schema `analytics` sama sekali. Lihat `eb_src/terminal_context.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from partrisk import config, feature_builder

from src import categorical_support  # survival_model/src (parent) - lihat README teknis soal path

CATEGORICAL_FEATURES = [
    "part_model_category",
    "client_category",
    "installation_age_band",
    "item_type_at_install_grouped",
    "terminal_type_grouped",
]
FINAL_TERMINAL_THRESHOLD = 200
_DROPPED_PREVIOUS_CYCLE = ["log_previous_cycle_lifetime_mean", "has_previous_cycle"]
_CONFIRMED_FAILURE_COLUMNS = [
    "log_previous_cycle_confirmed_failure_lifetime_mean",
    "has_previous_cycle_confirmed_failure_lifetime_mean",
]
# NUMERIC_FEATURES = seluruh config.NUMERIC_FEATURES (TERMASUK
# log_days_since_installation - TIDAK di-drop di sini, lihat docstring modul)
# minus previous_cycle_lifetime_mean lama (diganti confirmed-failure-only,
# validasi sama seperti model statis - reports/previous_cycle_audit.md).
# Kolom dari eb_src/dynamic_history.py (degradation trend + cumulative usage
# + jendela corrective tambahan) - urutan HARUS sama dengan urutan kolom
# yang dikembalikan tiap fungsi (lihat attach_dynamic_extra() di bawah).
DYNAMIC_EXTRA_NUMERIC_COLUMNS = [
    "log_failure_interval_mean_days", "log_failure_interval_last_days",
    "failure_interval_trend_ratio", "has_failure_interval_trend",
    "log_cumulative_prior_cycle_days", "log_physical_age_now", "previous_cycle_count",
    "prior_corrective_60d", "log_prior_corrective_60d", "prior_corrective_90d", "log_prior_corrective_90d",
    # Jendela 7/14 hari DICOBA lalu DIBATALKAN (regresi pada retrain penuh
    # dengan database fresh - lihat catatan di dynamic_history.
    # windowed_corrective_extra() dan reports/short_window.md).
]
NUMERIC_FEATURES = (
    [c for c in config.NUMERIC_FEATURES if c not in _DROPPED_PREVIOUS_CYCLE]
    + _CONFIRMED_FAILURE_COLUMNS + DYNAMIC_EXTRA_NUMERIC_COLUMNS
)
FLEET_FEATURES = list(config.FLEET_FEATURES)
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES + FLEET_FEATURES

FINAL_CATEGORY_THRESHOLDS = {"item_model_code_clean": 200, "item_type_at_install": 300}


def point_in_time_support(
    landmarks: pd.DataFrame, baseline_installs: pd.DataFrame, group_column: str
) -> pd.Series:
    """Dukungan point-in-time PER LANDMARK: jumlah LIFECYCLE (bukan baris
    landmark) pada `group_column` yang sama dengan `installed_on <=
    observation_on` landmark ini.

    `baseline_installs` HARUS satu baris per lifecycle (installed_on,
    group_column) dari cohort PENUH (populasi sama dengan dukungan model
    statis - `is_initial_model_cohort`, BUKAN dibatasi ke landmark eligible)
    - lihat catatan "TIDAK reuse" di docstring modul untuk alasannya.
    """
    keys = landmarks[group_column].fillna(config.UNKNOWN_LABEL).astype(str)
    base_keys = baseline_installs[group_column].fillna(config.UNKNOWN_LABEL).astype(str)
    base_installed = baseline_installs["installed_on"].to_numpy("datetime64[ns]")
    at = landmarks["observation_on"].to_numpy("datetime64[ns]")

    result = np.zeros(len(landmarks), dtype="int64")
    grouped_base = pd.Series(base_installed, index=base_keys.to_numpy()).groupby(level=0)
    for key, rows in keys.groupby(keys, sort=False).indices.items():
        if key not in grouped_base.groups:
            continue
        times_sorted = np.sort(grouped_base.get_group(key).to_numpy())
        # side="right": dukungan MENCAKUP lifecycle yang installed_on-nya
        # PERSIS di landmark ini (konsisten dengan feature_builder.
        # cumulative_support, yang juga inklusif terhadap baris sendiri).
        result[rows] = np.searchsorted(times_sorted, at[rows], side="right")
    return pd.Series(result, index=landmarks.index)


def attach_dynamic_extra(landmarks: pd.DataFrame, cycles: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Tempelkan `DYNAMIC_EXTRA_NUMERIC_COLUMNS` ke `landmarks` - dipanggil
    SETELAH `feature_builder.attach_history`/`attach_fleet`, SEBELUM
    `compute_features()`. `cycles`/`events` = populasi PENUH dari
    `data_reader.get_cycles()`/`get_events()` (sama seperti seluruh fitur
    lain di modul ini)."""
    from . import dynamic_history  # import lokal - hindari import siklik modul (dynamic_history tidak impor features)

    cum = dynamic_history.cumulative_cycle_age(cycles)
    landmarks = landmarks.merge(cum, on="installation_cycle_id", how="left")
    physical_age_now = landmarks["cumulative_prior_cycle_days"].to_numpy() + landmarks["landmark_age_days"].to_numpy()
    landmarks["log_cumulative_prior_cycle_days"] = np.log1p(landmarks["cumulative_prior_cycle_days"].to_numpy())
    landmarks["log_physical_age_now"] = np.log1p(np.clip(physical_age_now, 0, None))
    landmarks["previous_cycle_count"] = landmarks["previous_cycle_count"].astype(float)

    trend = dynamic_history.corrective_degradation_trend(landmarks, events)
    windowed = dynamic_history.windowed_corrective_extra(landmarks, events)
    return pd.concat([landmarks.reset_index(drop=True), trend, windowed], axis=1)


def attach_terminal_extra(landmarks: pd.DataFrame, terminal_raw: pd.DataFrame) -> pd.DataFrame:
    """Tempelkan `terminal_type_context` (KONSTAN per lifecycle, point-in-time
    filtered - lihat `terminal_context.py`) ke `landmarks`. `terminal_raw` =
    `data_reader.get_terminal_context()` APA ADANYA. Dukungan/grouping-nya
    SENGAJA TIDAK dihitung di sini (lihat `compute_features()`) - pola yang
    SAMA dengan part_model_category/item_type_at_install_grouped: dukungan
    HARUS dibekukan saat training dan dipakai ulang saat prediction (bukan
    dihitung ulang dari satu baris, yang akan selalu memberi dukungan=1)."""
    from . import terminal_context

    return terminal_context.attach_terminal_context(landmarks, terminal_raw)


def compute_features(
    landmarks: pd.DataFrame, support: pd.Series, item_type_support: pd.Series, terminal_support: pd.Series
) -> pd.DataFrame:
    """Fitur FINAL event-based. `landmarks` harus sudah melalui
    `feature_builder.attach_history`/`attach_fleet` (observation_on=landmark),
    `install_context.attach_install_context`, `attach_terminal_extra`, DAN
    merge kolom confirmed-failure previous-cycle (`_CONFIRMED_FAILURE_COLUMNS`)
    - lihat build_dataset.py untuk urutan pemanggilan lengkap.

    `support`/`item_type_support`/`terminal_support` dari
    `point_in_time_support()` di atas - HARUS dihitung sebelum dipanggil
    (bukan di dalam fungsi ini), sama seperti pola `features.compute_features()`
    model statis (dibekukan saat training, dipakai ulang saat prediction -
    lihat predict.py).
    """
    # feature_builder.build_features() sendiri menghitung part_model_category
    # pakai config.MIN_PART_MODEL_SUPPORT=300 (threshold classification,
    # dikalibrasi skala 251rb baris) - BUKAN threshold 200 yang tervalidasi
    # untuk skala survival (lihat FINAL_CATEGORY_THRESHOLDS di atas dan
    # README model statis poin 4/8 soal jebakan yang SAMA PERSIS pernah
    # terjadi di sana). Jadi part_model_category dihitung SENDIRI di sini
    # lewat categorical_support.apply_threshold + `support` milik fungsi ini
    # (point_in_time_support, threshold=200) - full["part_model_category"]
    # dari feature_builder TIDAK dipakai sama sekali.
    full = feature_builder.build_features(landmarks, support)

    result = pd.DataFrame(index=landmarks.index)
    result["part_model_category"] = categorical_support.apply_threshold(
        landmarks["item_model_code_clean"], support, FINAL_CATEGORY_THRESHOLDS["item_model_code_clean"]
    ).to_numpy()
    result["client_category"] = full["client_category"].to_numpy()
    result["installation_age_band"] = full["installation_age_band"].to_numpy()
    result["item_type_at_install_grouped"] = categorical_support.apply_threshold(
        landmarks["item_type_at_install"], item_type_support, FINAL_CATEGORY_THRESHOLDS["item_type_at_install"]
    ).to_numpy()
    result["terminal_type_grouped"] = categorical_support.apply_threshold(
        landmarks["terminal_type_context"], terminal_support, FINAL_TERMINAL_THRESHOLD
    ).to_numpy()

    for column in [c for c in config.NUMERIC_FEATURES if c not in _DROPPED_PREVIOUS_CYCLE] + FLEET_FEATURES:
        result[column] = full[column].to_numpy()
    for column in _CONFIRMED_FAILURE_COLUMNS:
        result[column] = landmarks[column].to_numpy()
    # DYNAMIC_EXTRA_NUMERIC_COLUMNS sudah ditempel attach_dynamic_extra()
    # LANGSUNG ke landmarks (bukan dari feature_builder.build_features(),
    # yang tidak tahu apa-apa soal kolom ini).
    for column in DYNAMIC_EXTRA_NUMERIC_COLUMNS:
        result[column] = landmarks[column].to_numpy()

    return result[FEATURE_COLUMNS].reset_index(drop=True)


def fit_encoder(train_features: pd.DataFrame, categorical_columns: list[str] | None = None) -> OneHotEncoder:
    columns = categorical_columns if categorical_columns is not None else CATEGORICAL_FEATURES
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train_features[columns])
    encoder.feature_names_used_ = list(columns)
    return encoder


def encode(
    features: pd.DataFrame, encoder: OneHotEncoder, numeric_columns: list[str] | None = None
) -> pd.DataFrame:
    columns = list(getattr(encoder, "feature_names_used_", CATEGORICAL_FEATURES))
    dummy_values = encoder.transform(features[columns])
    dummy_columns = encoder.get_feature_names_out(columns)
    dummies = pd.DataFrame(dummy_values, columns=dummy_columns, index=features.index)
    numeric = numeric_columns if numeric_columns is not None else NUMERIC_FEATURES + FLEET_FEATURES
    numeric_frame = features[numeric].astype(float)
    return pd.concat([numeric_frame, dummies], axis=1)
