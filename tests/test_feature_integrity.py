"""Konsistensi kolom fitur, pemuatan model, dan perlindungan kebocoran masa depan.

Tiga hal yang kalau rusak GAGAL DIAM-DIAM - tidak ada error, hanya prediksi
yang diam-diam salah:

- Kolom fitur yang dibangun harus persis sama dengan yang dipelajari model
  (urutan dan nama) - CatBoost/sklearn tidak memvalidasi nama kolom secara
  ketat, jadi pergeseran urutan bisa lolos tanpa exception.
- Model harus dimuat SEKALI per proses dan dipakai ulang - bukan dimuat ulang
  diam-diam setiap request (mahal), dan proses yang sudah hidup harus tetap
  bisa mengambil model versi baru begitu di-restart.
- Fitur riwayat tidak boleh melihat event yang terjadi SETELAH titik
  observasi - ini fondasi supervised learning yang valid untuk model ini.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
import feature_builder
import predict as failure_model
import scrap_features
import train
from tests.conftest import needs_database, needs_models

# ---------------------------------------------------------------------------
# Konsistensi kolom fitur
# ---------------------------------------------------------------------------


def _minimal_failure_raw(n: int = 3) -> pd.DataFrame:
    """Baris mentah minimal yang cukup untuk build_features() model kerusakan."""
    return pd.DataFrame({
        "item_model_code_clean": ["0521201"] * n,
        "installed_client_clean": ["CLIENT A"] * n,
        "days_since_installation": [10.0, 200.0, 800.0][:n],
        "total_prior_events": [1, 2, 3][:n],
        "prior_failure_count": [0, 1, 0][:n],
        "prior_corrective_count": [0, 1, 0][:n],
        "days_since_last_corrective": [np.nan, 5.0, np.nan][:n],
        "prior_distinct_places": [1, 2, 1][:n],
        "prior_corrective_30d": [0, 1, 0][:n],
        "prior_failure_365d": [0, 1, 0][:n],
        "prior_events_180d": [1, 2, 1][:n],
        "previous_cycle_lifetime_mean": [np.nan, 100.0, np.nan][:n],
        "has_previous_cycle": [False, True, False][:n],
        "observation_on": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"][:n]),
        "log_model_failures_90d": [0.0, 1.0, 0.5][:n],
        "model_failure_rate_90d": [0.0, 0.1, 0.05][:n],
        "log_model_fleet_size": [1.0, 2.0, 1.5][:n],
    })


def test_kolom_fitur_kerusakan_persis_sama_dengan_config():
    """CatBoost tidak memvalidasi nama kolom secara ketat - pergeseran urutan
    bisa lolos tanpa exception dan diam-diam salah membaca fitur."""
    raw = _minimal_failure_raw()
    support = pd.Series([500] * len(raw))
    features = feature_builder.build_features(raw, support)
    assert list(features.columns) == config.FEATURE_COLUMNS


def test_kolom_fitur_kerusakan_konsisten_lewat_project_features():
    """project_features() (dipakai prediction multi-horizon) harus tetap
    menghasilkan kolom yang sama - bukan hanya build_features() langsung."""
    raw = _minimal_failure_raw()
    support = pd.Series([500] * len(raw))
    for step in range(4):
        features = feature_builder.project_features(raw, support, step)
        assert list(features.columns) == config.FEATURE_COLUMNS, f"step={step}"


def test_kolom_fitur_scrap_persis_sama_dengan_config():
    episodes = pd.DataFrame({
        "item_type_clean": ["MOTOR", "PC", None],
        "age_total_days": [100.0, 500.0, 10.0],
        "cycle_age_days": [50.0, 100.0, 5.0],
        "prior_repaired_count": [0, 2, 0],
        "prior_failure_count": [1, 3, 1],
    })
    features = scrap_features.build_features(episodes, known_types=["MOTOR", "PC"])
    assert list(features.columns) == config.SCRAP_FEATURE_COLUMNS


def test_urutan_kolom_kategorikal_dan_numerik_tidak_bercampur():
    """CATEGORICAL_FEATURES harus muncul lebih dulu di FEATURE_COLUMNS -
    train_model() memakai config.CATEGORICAL_FEATURES sebagai daftar terpisah
    untuk cat_features Pool(); kalau urutannya tidak konsisten dengan
    FEATURE_COLUMNS, cat_features bisa menunjuk indeks kolom yang salah."""
    n_cat = len(config.CATEGORICAL_FEATURES)
    assert config.FEATURE_COLUMNS[:n_cat] == config.CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# Pemuatan dan versi model
# ---------------------------------------------------------------------------


@needs_models
def test_model_dimuat_sekali_dipakai_ulang():
    """Dua panggilan _load_model() dalam proses yang sama harus mengembalikan
    OBJEK yang sama - bukan memuat ulang dari disk setiap kali dipanggil."""
    failure_model._LOADED = None
    first = failure_model._load_model()
    second = failure_model._load_model()
    assert first[0] is second[0]
    assert first[1] is second[1]


@needs_models
def test_model_baru_terlihat_setelah_cache_direset(tmp_path_factory=None):
    """Proses yang di-restart (di sini disimulasikan dengan mereset cache
    modul) harus mengambil model yang ditunjuk CURRENT saat itu - bukan
    versi lama yang kebetulan tersimpan di memori.

    Ini bukan uji ganti CURRENT sungguhan (itu tanggung jawab train.py) -
    hanya membuktikan _load_model() tidak punya cache yang bocor lintas
    "proses" (di sini: lintas reset)."""
    failure_model._LOADED = None
    first = failure_model._load_model()
    version_first = first[2]["model_version"]

    failure_model._LOADED = None
    second = failure_model._load_model()
    version_second = second[2]["model_version"]

    assert version_first == version_second  # CURRENT belum berubah
    assert first[0] is not second[0]  # tapi objeknya BENAR-BENAR dimuat ulang


@needs_models
def test_fleet_snapshot_ikut_dibuang_saat_model_direset():
    """Potret armada milik predict.py terikat ke model yang sedang dimuat -
    reset cache model juga harus membuang potret lama, bukan meninggalkan
    potret dari model sebelumnya menempel ke model baru."""
    failure_model._LOADED = None
    failure_model._FLEET = None
    metadata = failure_model._load_model()[2]
    snapshot = failure_model._fleet_snapshot(pd.Timestamp(metadata["fleet_snapshot_at"]))
    assert not snapshot.empty
    assert set(config.FLEET_FEATURES) <= set(snapshot.columns)


# ---------------------------------------------------------------------------
# Perlindungan kebocoran masa depan
# ---------------------------------------------------------------------------


def _events(rows: list[tuple]) -> pd.DataFrame:
    """rows: (created_on, wo_type, is_failure, place)"""
    return pd.DataFrame({
        "item_identifier_clean": ["ITEM-A"] * len(rows),
        "created_on": pd.to_datetime([r[0] for r in rows]),
        "wo_type_clean": [r[1] for r in rows],
        "status_clean": ["DISMANTLED"] * len(rows),
        "is_failure_onset": [r[2] for r in rows],
        "place_canonical_clean": [r[3] for r in rows],
    })


def _observation(at: str) -> pd.DataFrame:
    return pd.DataFrame({
        "item_identifier_clean": ["ITEM-A"],
        "observation_on": pd.to_datetime([at]),
    })


def test_attach_history_tidak_melihat_event_setelah_observasi():
    """Menambahkan event SETELAH titik observasi tidak boleh mengubah hasil
    riwayat sama sekali - kalau berubah, berarti ada kebocoran masa depan."""
    baseline_events = _events([
        ("2026-01-01", "CORRECTIVE", True, "LOKASI A"),
        ("2026-01-15", "PREVENTIVE", False, "LOKASI A"),
    ])
    future_events = pd.concat([
        baseline_events,
        _events([("2026-06-01", "CORRECTIVE", True, "LOKASI B")]),  # setelah observasi
    ], ignore_index=True)

    observation = _observation("2026-02-01")
    baseline_result = feature_builder.attach_history(observation.copy(), baseline_events)
    with_future = feature_builder.attach_history(observation.copy(), future_events)

    for column in feature_builder._HISTORY_COUNTS:
        assert baseline_result[column].iloc[0] == with_future[column].iloc[0], column
    assert (
        baseline_result["days_since_last_corrective"].iloc[0]
        == with_future["days_since_last_corrective"].iloc[0]
    )


def test_attach_history_event_pada_detik_observasi_ikut_terhitung():
    """Event PERSIS pada observation_on harus ikut (batas <=), tetapi event
    sesudahnya tidak - ini batas yang menentukan, jadi diuji eksplisit."""
    at_boundary = _events([("2026-02-01", "CORRECTIVE", True, "LOKASI A")])
    after_boundary = _events([("2026-02-01 00:00:01", "CORRECTIVE", True, "LOKASI A")])

    observation = _observation("2026-02-01")
    included = feature_builder.attach_history(observation.copy(), at_boundary)
    excluded = feature_builder.attach_history(observation.copy(), after_boundary)

    assert included["prior_failure_count"].iloc[0] == 1
    assert excluded["prior_failure_count"].iloc[0] == 0


def test_project_features_tidak_memakai_kejadian_sungguhan_masa_depan():
    """project_features(steps_ahead>0) mensimulasikan waktu berjalan TANPA
    kejadian baru - hitungan riwayat harus BEKU, hanya umur yang bertambah."""
    raw = _minimal_failure_raw(n=1)
    support = pd.Series([500])

    baseline = feature_builder.project_features(raw, support, steps_ahead=0)
    projected = feature_builder.project_features(raw, support, steps_ahead=2)

    # Hitungan riwayat tidak berubah walau waktu "maju".
    assert (
        baseline["log_prior_failure_count"].iloc[0]
        == projected["log_prior_failure_count"].iloc[0]
    )
    assert (
        baseline["log_total_prior_events"].iloc[0]
        == projected["log_total_prior_events"].iloc[0]
    )
    # Tapi umur pemasangan BERTAMBAH sesuai waktu yang disimulasikan.
    assert (
        projected["log_days_since_installation"].iloc[0]
        > baseline["log_days_since_installation"].iloc[0]
    )


def test_label_negatif_dekat_batas_data_tidak_dipakai():
    """Snapshot NEGATIF (tidak pernah rusak) yang 30 hari ke depannya belum
    sepenuhnya terekam di data_end tidak boleh dipakai - belum bisa
    dipastikan itu benar-benar negatif atau sekadar belum sempat rusak.

    Label POSITIF tidak butuh proteksi ini: kerusakan yang sudah tercatat
    adalah fakta, tidak peduli seberapa dekat dengan data_end.
    """
    last_confirmable = pd.Timestamp("2026-07-04")  # data_end - horizon, dari SQL
    cycle = pd.DataFrame({
        "is_initial_model_cohort": [True],
        "installed_on": pd.to_datetime(["2026-01-01"]),
        "cycle_end_on": pd.to_datetime(["2026-08-03"]),  # masih berjalan (censored)
        "dataset_max_event_on": pd.to_datetime(["2026-08-03"]),
        "failure_onset_on": [pd.NaT],  # tidak pernah rusak
        "is_recon_verified_negative_eligible": [True],
        "last_confirmable_observation_on": [last_confirmable],
    })

    observations = feature_builder.training_observations(cycle)

    before_boundary = observations["observation_on"] <= last_confirmable
    after_boundary = observations["observation_on"] > last_confirmable

    assert before_boundary.any() and after_boundary.any(), (
        "grid observasi test tidak mencakup kedua sisi batas - perbaiki setup test"
    )
    assert observations.loc[before_boundary, "is_eligible"].all(), (
        "observasi sebelum batas confirmable seharusnya layak dipakai (negatif terbukti)"
    )
    assert not observations.loc[after_boundary, "is_eligible"].any(), (
        "observasi SETELAH batas confirmable terpakai - hasilnya belum tentu benar "
        "negatif, ini kebocoran: model bisa belajar dari label yang belum pasti"
    )


def test_label_positif_dekat_batas_data_tetap_dipakai():
    """Kerusakan yang SUDAH tercatat adalah fakta - tidak perlu menunggu
    konfirmasi seperti label negatif, walau observasinya dekat data_end."""
    cycle = pd.DataFrame({
        "is_initial_model_cohort": [True],
        "installed_on": pd.to_datetime(["2026-07-01"]),
        "cycle_end_on": pd.to_datetime(["2026-07-20"]),
        "dataset_max_event_on": pd.to_datetime(["2026-07-20"]),
        "failure_onset_on": pd.to_datetime(["2026-07-20"]),  # rusak di akhir siklus
        "is_recon_verified_negative_eligible": [False],  # tidak relevan untuk positif
        "last_confirmable_observation_on": pd.to_datetime(["2026-06-01"]),  # jauh sebelum
    })

    observations = feature_builder.training_observations(cycle)

    assert observations["target_failure"].any(), "setup test tidak menghasilkan target positif"
    positive_rows = observations.loc[observations["target_failure"]]
    assert positive_rows["is_eligible"].all(), (
        "observasi dengan kerusakan terkonfirmasi tidak boleh dibuang hanya karena "
        "dekat batas data - kerusakan yang tercatat adalah fakta, bukan estimasi"
    )


def test_assign_split_membuang_observasi_terlalu_lama():
    """Observasi sebelum MIN_OBSERVATION_DATE tidak boleh ikut dilatih -
    periode itu dianggap tidak cukup representatif (lihat config.py)."""
    data_end = pd.Timestamp("2026-08-03")
    dataset = pd.DataFrame({
        "observation_on": pd.to_datetime([
            "2010-01-01",  # sebelum MIN_OBSERVATION_DATE (2014-01-01)
            "2020-01-01",  # jauh di masa lalu, sudah pasti resolved -> TRAIN
        ]),
    })
    split = train.assign_split(dataset, data_end)
    assert split.iloc[0] == "EXCLUDED_TOO_OLD"
    assert split.iloc[1] == train.TRAIN


def test_assign_split_test_dan_validation_tidak_tumpang_tindih_dengan_train():
    """Batas antar-split memakai tanggal RESOLVED (observation_on + horizon),
    bukan observation_on mentah - supaya baris TRAIN tidak diam-diam
    mengandung jawaban yang sebenarnya baru terungkap di periode VALIDATION."""
    data_end = pd.Timestamp("2026-08-03")
    horizon = pd.Timedelta(days=config.TARGET_HORIZON_DAYS)
    validation_start = pd.Timestamp(year=data_end.year, month=1, day=1) - pd.DateOffset(years=1)

    # Observation_on sebelum validation_start, tetapi jawabannya baru
    # terungkap SESUDAH validation_start dimulai (resolved >= validation_start).
    dataset = pd.DataFrame({
        "observation_on": [validation_start - pd.Timedelta(days=1)],
    })
    split = train.assign_split(dataset, data_end)
    assert split.iloc[0] != train.TRAIN, (
        "observasi yang jawabannya baru terungkap di periode VALIDATION "
        "tidak boleh ikut TRAIN - itu kebocoran dari masa depan (relatif "
        "terhadap TRAIN) ke masa lalu"
    )
