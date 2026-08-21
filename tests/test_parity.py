"""Batch scoring harus menghasilkan angka yang SAMA dengan prediksi satu PART.

Ini test terpenting di repository ini.

Batch dan single memakai jalur kode yang berbeda - batch membaca seluruh
database sekali lalu memvektorkan, single membaca riwayat satu PART - jadi
selalu ada kemungkinan keduanya diam-diam menyimpang. Kalau itu terjadi,
daftar prioritas di dashboard tidak lagi cocok dengan angka yang dilihat
teknisi pada halaman detail, dan tidak ada yang akan menyadarinya.

Angkanya dibandingkan PERSIS (bukan sekadar mirip): kedua jalur memanggil
fungsi feature_builder dan model yang sama, jadi hasilnya seharusnya identik
bit demi bit, bukan sekadar dekat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from partrisk import config, scrap_features
from partrisk.data import reader as data_reader
from partrisk.predict import failure as failure_model
from partrisk.predict import scrap as scrap_model
from partrisk.serving import batch_predictor
from tests.conftest import needs_database, needs_models

pytestmark = [needs_database, needs_models]

# Diambil dari ujung atas, tengah, dan ujung bawah daftar supaya perbedaan
# yang hanya muncul pada PART berisiko rendah tetap tertangkap.
SAMPLE_SIZE = 6


@pytest.fixture(scope="module")
def sample(batch) -> pd.DataFrame:
    frame = batch.frame
    positions = np.unique(
        np.linspace(0, len(frame) - 1, SAMPLE_SIZE).astype(int)
    )
    return frame.iloc[positions]


def test_probabilitas_kerusakan_batch_sama_dengan_single(sample):
    for _, row in sample.iterrows():
        single = failure_model.predict(row["item_id"])
        for days in config.PREDICTION_HORIZON_DAYS:
            column = f"failure_probability_{days}d"
            assert single[column] == row[column], (
                f"{row['item_id']} horizon {days}d: "
                f"single={single[column]} batch={row[column]}"
            )


def test_kelompok_risiko_kerusakan_batch_sama_dengan_single(sample):
    for _, row in sample.iterrows():
        single = failure_model.predict(row["item_id"])
        assert single["risk_level"] == row["failure_risk_level"]


def test_probabilitas_scrap_batch_sama_dengan_single(sample):
    for _, row in sample.iterrows():
        single = scrap_model.predict_scrap(row["item_id"])
        assert single["scrap_probability"] == row["scrap_probability"]
        assert single["scrap_risk_level"] == row["scrap_risk_level"]
        assert single["item_type"] == row["item_type"]


def test_kolom_mentah_scrap_batch_sama_dengan_current_state(batch, sample):
    """Penyusun kolom scrap versi batch harus setara current_state().

    Ini satu-satunya bagian yang ditulis ulang untuk batch, jadi diperiksa
    kolom per kolom - bukan hanya hasil akhirnya.
    """
    items = sample["item_id"]
    cycles = data_reader.get_cycles()
    events = data_reader.get_events()
    batched = batch_predictor._scrap_states(
        events, cycles, batch.data_end, items
    ).set_index("item_identifier_clean")

    for item in items:
        single = scrap_features.current_state(
            data_reader.get_events(item),
            data_reader.get_cycles(item, batch.data_end),
            batch.data_end,
        )
        assert not single.empty, item
        expected = single.iloc[0]
        actual = batched.loc[item]
        for column in (
            "item_type_clean",
            "age_total_days",
            "cycle_age_days",
            "prior_repaired_count",
            "prior_failure_count",
            "failure_onset_on",
        ):
            left, right = expected[column], actual[column]
            if isinstance(left, float) and np.isnan(left):
                assert np.isnan(right), f"{item}.{column}"
            else:
                assert left == right, f"{item}.{column}: {left!r} != {right!r}"


def test_populasi_batch_sama_dengan_yang_dipakai_menyetel_ambang(batch):
    """Jumlah PART aktif dan jumlah HIGH harus cocok dengan metadata training.

    train.py menyetel ambang HIGH dari kapasitas kerja bulanan dengan menskor
    seluruh PART aktif, dan mencatat hasilnya di metadata. Kalau batch di sini
    menghasilkan populasi yang berbeda, berarti salah satu jalur berubah.
    """
    metadata = failure_model._load_model()[2]
    basis = metadata["cutoff_basis"]
    if metadata["fleet_snapshot_at"] != str(batch.data_end):
        pytest.skip("database sudah bertambah sejak model dilatih")

    assert len(batch.frame) == basis["active_parts_scored"]
    high = int(batch.frame["failure_risk_level"].eq("HIGH").sum())
    assert high == basis["flagged_high"]


def test_urutan_prioritas_konsisten_dengan_kelompok_risiko(batch):
    """Tidak boleh ada PART LOW yang berperingkat di atas PART HIGH."""
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    ranks = batch.frame["failure_risk_level"].map(order).to_numpy()
    assert np.all(np.diff(ranks) >= 0)


def test_risiko_kumulatif_tidak_pernah_menurun(batch):
    """Risiko 30d <= 60d <= 90d <= 120d, dijamin oleh perantaian hazard."""
    horizons = config.PREDICTION_HORIZON_DAYS
    for earlier, later in zip(horizons, horizons[1:]):
        assert (
            batch.frame[f"failure_probability_{earlier}d"]
            <= batch.frame[f"failure_probability_{later}d"] + 1e-12
        ).all()
