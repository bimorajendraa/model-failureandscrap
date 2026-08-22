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
from partrisk.predict import survival as predict_survival
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


def test_median_days_to_failure_batch_sama_dengan_single(sample):
    """Field advisory model survival (mode aditif, gate_decision.md) - batch
    (score_batch(), divektorkan) harus setuju dengan single (predict(),
    per-PART) untuk PART yang sama, sama seperti pasangan CatBoost di atas.
    Skip diam-diam kalau model survival belum pernah dilatih di mesin ini -
    field advisory ini tidak wajib ada (lihat _survival_advisory_fields())."""
    try:
        predict_survival.load_model()
    except FileNotFoundError:
        pytest.skip("model survival belum dilatih (survival_model/event_based/artifacts/)")

    for _, row in sample.iterrows():
        single = predict_survival.predict(row["item_id"])
        for batch_column, single_key in (
            ("median_days_to_failure", "median_days_remaining_from_now"),
            ("days_until_survival_90pct", "days_until_survival_90pct_from_now"),
            ("days_until_risk_medium", "days_until_risk_medium_from_now"),
            ("days_until_risk_high", "days_until_risk_high_from_now"),
        ):
            expected, actual = single[single_key], row[batch_column]
            if expected is None:
                assert pd.isna(actual), f"{row['item_id']}.{batch_column}: single=None batch={actual}"
            else:
                assert expected == actual, (
                    f"{row['item_id']}.{batch_column}: single={expected} batch={actual}"
                )


def test_survival_kurva_terkalibrasi_monoton_turun_dan_flag_benar(sample):
    """Fase upgrade RSF, Langkah B: median/p90/kurva sekarang dari kurva
    TERKALIBRASI (curves.calibrate_curve()), bukan kurva mentah lagi - lihat
    reports/rsf_median_curve_calibration_result.md. curve_is_calibrated harus
    True kalau calibrators.joblib ada (artifact production sekarang selalu
    punya file ini sejak Fase A3), dan kurva yang dikembalikan harus tetap
    S(t) valid (monoton turun, dalam [0,1]) - kalibrasi per-titik + interpolasi
    TIDAK boleh merusak properti dasar kurva survival."""
    try:
        predict_survival.load_model()
    except FileNotFoundError:
        pytest.skip("model survival belum dilatih (survival_model/event_based/artifacts/)")

    for _, row in sample.iterrows():
        result = predict_survival.predict(row["item_id"])
        assert result["curve_is_calibrated"] is True
        curve = result["estimated_survival_curve_from_now"]
        if not curve:
            continue
        probs = [point["survival_probability"] for point in curve]
        assert all(0.0 - 1e-9 <= p <= 1.0 + 1e-9 for p in probs)
        assert all(a >= b - 1e-9 for a, b in zip(probs, probs[1:]))


def test_survival_urutan_ambang_waktu_konsisten(sample):
    """Langkah C: days_until_survival_90pct (S<=0,9) harus tercapai LEBIH
    DULU (hari lebih kecil) daripada days_until_risk_medium (S<=0,85),
    yang harus lebih dulu dari days_until_risk_high (S<=0,75) - S(t) monoton
    turun, jadi ambang yang lebih dalam PASTI butuh waktu >= ambang yang
    lebih dangkal. Hanya dibandingkan kalau KEDUANYA terisi (bukan None)."""
    try:
        predict_survival.load_model()
    except FileNotFoundError:
        pytest.skip("model survival belum dilatih (survival_model/event_based/artifacts/)")

    checked_any = False
    for _, row in sample.iterrows():
        result = predict_survival.predict(row["item_id"])
        p90 = result["days_until_survival_90pct_from_now"]
        medium = result["days_until_risk_medium_from_now"]
        high = result["days_until_risk_high_from_now"]
        if p90 is not None and medium is not None:
            checked_any = True
            assert p90 <= medium + 1e-9, (row["item_id"], p90, medium)
        if medium is not None and high is not None:
            checked_any = True
            assert medium <= high + 1e-9, (row["item_id"], medium, high)
    if not checked_any:
        pytest.skip("tidak ada sample dengan pasangan ambang terisi untuk diuji")


def test_survival_calibrated_risk_monoton_naik(sample):
    """Fase R1 upgrade RSF: risk per horizon dikalibrasi isotonic SENDIRI-
    SENDIRI per horizon (fit_calibrators()), yang BISA saling silang walau
    S(t) mentahnya monoton turun - _calibrate_risk() WAJIB menegakkan cummax
    30->60->90->120. Test ini menegakkan invariant itu tetap benar di jalur
    production (predict()), bukan cuma di unit test fungsi kalibrasinya
    sendiri - pola sama dengan monotonisitas kalibrasi CatBoost."""
    try:
        predict_survival.load_model()
    except FileNotFoundError:
        pytest.skip("model survival belum dilatih (survival_model/event_based/artifacts/)")

    checked_any = False
    for _, row in sample.iterrows():
        result = predict_survival.predict(row["item_id"])
        values = [result[f"calibrated_risk_{h}d"] for h in predict_survival.HORIZONS_DAYS]
        if any(v is None for v in values):
            continue  # beyond_training_followup atau calibrators None - tidak ada yang ditegakkan
        checked_any = True
        for a, b in zip(values, values[1:]):
            assert a <= b + 1e-9, f"{row['item_id']}: calibrated_risk turun {values}"
    if not checked_any:
        pytest.skip("tidak ada sample dengan calibrated_risk terisi untuk diuji")


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
