"""Fitur baseline instalasi untuk survival model - membungkus feature_builder
apa adanya, tidak menghitung ulang logic yang sudah ada.

Setiap lifecycle diobservasi PERSIS pada `installed_on` (awal siklus). Ini
sengaja beda dari model classification yang meng-observasi tiap 30 hari
sepanjang siklus - lihat README.md bagian "Keterbatasan: baseline instalasi
vs kondisi sekarang". Konsekuensinya, `log_days_since_installation` dan
`installation_age_band` (2 dari 21 fitur classification) SELALU bernilai
konstan (umur=0) di sini, karena umur pemasangan justru menjadi SUMBU WAKTU
model survival (`duration_days`), bukan fitur input - keduanya di-drop, 19
fitur classification lainnya dipakai apa adanya.
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

import config
import feature_builder

DROPPED_AT_INSTALL_FEATURES = ["log_days_since_installation", "installation_age_band"]

CATEGORICAL_FEATURES = [c for c in config.CATEGORICAL_FEATURES if c not in DROPPED_AT_INSTALL_FEATURES]
NUMERIC_FEATURES = [c for c in config.NUMERIC_FEATURES if c not in DROPPED_AT_INSTALL_FEATURES]
FLEET_FEATURES = list(config.FLEET_FEATURES)
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES + FLEET_FEATURES


def build_baseline_observations(cycles: pd.DataFrame) -> pd.DataFrame:
    """Satu baris per lifecycle, observation_on = installed_on. Umur
    pemasangan diset 0 secara langsung (bukan lewat pengurangan tanggal
    generik) karena secara konstruksi observation_on == installed_on selalu."""
    observations = cycles.reset_index(drop=True).copy()
    observations["observation_on"] = observations["installed_on"]
    observations["days_since_installation"] = 0.0
    return observations


def attach_survival_features(
    observations: pd.DataFrame, events: pd.DataFrame, cycles: pd.DataFrame, episodes: pd.DataFrame
) -> pd.DataFrame:
    """Riwayat + kondisi armada point-in-time terhadap installed_on. Reuse
    langsung feature_builder.attach_history/attach_fleet - keduanya sudah
    aman (hanya memakai event <= observation_on), tidak ada logic baru."""
    observations = feature_builder.attach_history(observations, events)
    observations = feature_builder.attach_fleet(observations, cycles, episodes)
    return observations


def compute_features(observations: pd.DataFrame, support: pd.Series) -> pd.DataFrame:
    """19 fitur baseline instalasi, lewat feature_builder.build_features()
    (reuse penuh) lalu diseleksi ke FEATURE_COLUMNS - membuang 2 kolom umur
    yang konstan di baseline instalasi (lihat docstring modul)."""
    full = feature_builder.build_features(observations, support)
    return full[FEATURE_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Encoding kategorikal: RSF/CoxPH (scikit-survival) tidak punya native
# categorical handling seperti CatBoost, jadi part_model_category/
# client_category di-one-hot. Encoder di-fit HANYA di TRAIN dan disimpan
# bersama model supaya inference memakai mapping yang identik.
# ---------------------------------------------------------------------------


def fit_encoder(train_features: pd.DataFrame) -> OneHotEncoder:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train_features[CATEGORICAL_FEATURES])
    return encoder


def encode(features: pd.DataFrame, encoder: OneHotEncoder) -> pd.DataFrame:
    dummy_values = encoder.transform(features[CATEGORICAL_FEATURES])
    dummy_columns = encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    dummies = pd.DataFrame(dummy_values, columns=dummy_columns, index=features.index)
    numeric = features[NUMERIC_FEATURES + FLEET_FEATURES].astype(float)
    return pd.concat([numeric, dummies], axis=1)
