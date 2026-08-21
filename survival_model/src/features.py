"""Fitur baseline instalasi untuk survival model.

Setiap lifecycle diobservasi PERSIS pada `installed_on` (awal siklus). Ini
sengaja beda dari model classification yang meng-observasi tiap 30 hari
sepanjang siklus - lihat README.md bagian "Keterbatasan: baseline instalasi
vs kondisi sekarang". Konsekuensinya, `log_days_since_installation` dan
`installation_age_band` (2 dari 21 fitur classification) SELALU bernilai
konstan (umur=0) di sini, karena umur pemasangan justru menjadi SUMBU WAKTU
model survival (`duration_days`), bukan fitur input - keduanya di-drop.

Fitur FINAL (dipakai train.py/evaluate.py/predict.py) adalah hasil audit
metodologis lengkap di `experiments.py` (lihat reports/category_threshold.md,
feature_ablation.md, previous_cycle_audit.md, model_comparison.md untuk bukti
angkanya) - BUKAN sekadar 19 fitur classification warisan:

- `item_type_at_install` (konteks instalasi, dari event INSTALLED yang SUDAH
  dibaca - tidak ada query baru) TERBUKTI menambah signal VALIDATION nyata
  di atas 19 fitur lama. `place_at_install` TIDAK - bahkan sedikit menurunkan
  skor saat digabung - jadi TIDAK diikutkan di final meski sempat diuji.
- `previous_cycle_lifetime_mean` (fitur lama) TERBUKTI mencampur durasi
  siklus sebelumnya apa pun cara berakhirnya (FAILURE/censored/reinstall) -
  diganti versi confirmed-failure-only yang lebih jujur DAN validasinya
  lebih baik.
- Threshold kategori KHUSUS survival (bukan `config.MIN_PART_MODEL_SUPPORT`
  classification, yang dikalibrasi untuk skala berbeda): part_model=200,
  item_type_at_install=300 - dipilih dari VALIDATION, lihat
  reports/category_threshold.md.
- Hyperparameter RSF: default/current TERBUKTI sudah optimal pada pencarian
  kecil (reports/model_comparison.md) - tidak berubah dari sesi sebelumnya.

`LEGACY_*` di bawah adalah 19 fitur classification warisan APA ADANYA -
dipertahankan hanya sebagai referensi/baseline pembanding
(`experiments.py` konfigurasi "A_current"), TIDAK dipakai model production.
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from partrisk import config, feature_builder

from . import categorical_support, install_context, previous_cycle

DROPPED_AT_INSTALL_FEATURES = ["log_days_since_installation", "installation_age_band"]

LEGACY_CATEGORICAL_FEATURES = [c for c in config.CATEGORICAL_FEATURES if c not in DROPPED_AT_INSTALL_FEATURES]
LEGACY_NUMERIC_FEATURES = [c for c in config.NUMERIC_FEATURES if c not in DROPPED_AT_INSTALL_FEATURES]
FLEET_FEATURES = list(config.FLEET_FEATURES)
LEGACY_FEATURE_COLUMNS = LEGACY_CATEGORICAL_FEATURES + LEGACY_NUMERIC_FEATURES + FLEET_FEATURES

# --- Konfigurasi FINAL (hasil experiments.py) --------------------------------
FINAL_CATEGORY_THRESHOLDS = {"item_model_code_clean": 200, "item_type_at_install": 300}
_DROPPED_PREVIOUS_CYCLE = ["log_previous_cycle_lifetime_mean", "has_previous_cycle"]
_CONFIRMED_FAILURE_COLUMNS = [
    "log_previous_cycle_confirmed_failure_lifetime_mean",
    "has_previous_cycle_confirmed_failure_lifetime_mean",
]

CATEGORICAL_FEATURES = ["part_model_category", "client_category", "item_type_at_install_grouped"]
NUMERIC_FEATURES = [
    c for c in LEGACY_NUMERIC_FEATURES if c not in _DROPPED_PREVIOUS_CYCLE
] + _CONFIRMED_FAILURE_COLUMNS
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


def attach_final_context(observations: pd.DataFrame, events: pd.DataFrame, cycles: pd.DataFrame) -> pd.DataFrame:
    """Tempelkan 2 penambahan yang TERBUKTI membantu VALIDATION di
    experiments.py: item_type_at_install (konteks instalasi) dan
    previous-cycle confirmed-failure-only. Panggil SETELAH
    attach_survival_features()."""
    observations = install_context.attach_install_context(observations, events)
    pc = previous_cycle.audit_previous_cycle_features(cycles)
    # transform_for_model() butuh KEDUA kolom audit (confirmed-failure-mean
    # DAN last-confirmed) untuk menghitung log1p/has_* keduanya - hanya
    # confirmed-failure-mean yang dipakai di fitur FINAL (last-confirmed
    # kalah tipis di previous_cycle_audit.md), sisanya diseleksi lewat
    # _CONFIRMED_FAILURE_COLUMNS di bawah.
    observations = observations.merge(
        pc[[
            "installation_cycle_id",
            "previous_cycle_confirmed_failure_lifetime_mean",
            "last_confirmed_failure_lifetime",
        ]],
        on="installation_cycle_id", how="left",
    )
    transform = previous_cycle.transform_for_model(observations)[_CONFIRMED_FAILURE_COLUMNS]
    return pd.concat([observations, transform], axis=1)


def compute_features(
    observations: pd.DataFrame, support: pd.Series, item_type_support: pd.Series | None = None
) -> pd.DataFrame:
    """Fitur FINAL production (lihat docstring modul). `observations` harus
    sudah melalui attach_survival_features() DAN attach_final_context().

    `support` = dukungan item_model_code_clean, `item_type_support` = dukungan
    item_type_at_install. Saat TRAINING (batch, banyak baris) keduanya
    dihitung point-in-time dari `observations` itu sendiri - `item_type_support`
    boleh dikosongkan (default None), dihitung otomatis di sini. Saat
    PREDICTION (satu baris/PART) keduanya HARUS diisi dari nilai yang
    DIBEKUKAN saat training (metadata.json) - menghitung ulang dari 1 baris
    akan selalu menghasilkan dukungan=1, salah total. Lihat predict.py."""
    full = feature_builder.build_features(observations, support)

    if item_type_support is None:
        item_type_support = categorical_support.cumulative_support(
            observations, "item_type_at_install", "observation_on"
        )

    result = pd.DataFrame(index=observations.index)
    result["part_model_category"] = categorical_support.apply_threshold(
        observations["item_model_code_clean"], support, FINAL_CATEGORY_THRESHOLDS["item_model_code_clean"]
    ).to_numpy()
    result["client_category"] = full["client_category"].to_numpy()
    result["item_type_at_install_grouped"] = categorical_support.apply_threshold(
        observations["item_type_at_install"], item_type_support, FINAL_CATEGORY_THRESHOLDS["item_type_at_install"]
    ).to_numpy()

    for column in [c for c in LEGACY_NUMERIC_FEATURES if c not in _DROPPED_PREVIOUS_CYCLE] + FLEET_FEATURES:
        result[column] = full[column].to_numpy()
    for column in _CONFIRMED_FAILURE_COLUMNS:
        result[column] = observations[column].to_numpy()

    return result[FEATURE_COLUMNS].reset_index(drop=True)


def item_type_support_totals(observations: pd.DataFrame) -> dict[str, int]:
    """Dukungan akhir item_type_at_install - dibekukan ke metadata model
    (pola yang sama dengan part_model_code_clean support_totals di
    feature_builder), dipakai predict.py supaya kategori yang dikenal model
    konsisten dengan saat ia dilatih."""
    support = categorical_support.cumulative_support(observations, "item_type_at_install", "observation_on")
    totals = observations.assign(_item_type_support=support.to_numpy())
    return categorical_support.support_totals(totals, "item_type_at_install")


# ---------------------------------------------------------------------------
# Encoding kategorikal: RSF/CoxPH (scikit-survival) tidak punya native
# categorical handling seperti CatBoost, jadi kolom kategorikal di-one-hot.
# Encoder di-fit HANYA di TRAIN dan disimpan bersama model supaya inference
# memakai mapping yang identik.
# ---------------------------------------------------------------------------


def fit_encoder(train_features: pd.DataFrame, categorical_columns: list[str] | None = None) -> OneHotEncoder:
    """`categorical_columns` default ke CATEGORICAL_FEATURES (fitur final
    production) - dioverride experiments.py untuk kombinasi kategorikal lain
    (context-only, combined, dst.) tanpa mengubah perilaku default."""
    columns = categorical_columns if categorical_columns is not None else CATEGORICAL_FEATURES
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train_features[columns])
    encoder.feature_names_used_ = list(columns)  # dipakai encode() supaya tidak perlu diulang tiap panggil
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
