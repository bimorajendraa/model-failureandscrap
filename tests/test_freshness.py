"""Kesegaran data dan penghematan pembacaan database.

Dua hal yang dijaga di sini pernah SALAH, dan keduanya gagal tanpa suara -
tidak ada error, prediksi tetap keluar, hanya angkanya yang keliru atau
pelayanannya berkali lipat lebih lambat. Justru karena itu perlu test.
"""

from __future__ import annotations

import pandas as pd
import pytest

import data_reader
import predict as failure_model
from api import data_state, query_cache
from api.services import prediction_service
from tests.conftest import needs_database, needs_models

pytestmark = [needs_database, needs_models]


@pytest.fixture
def count_connections(monkeypatch):
    """Hitung berapa koneksi database yang dibuka."""
    original = data_reader.connect
    counter = {"n": 0}

    def counted(*args, **kwargs):
        counter["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(data_reader, "connect", counted)
    return counter


# ---------------------------------------------------------------------------
# Potret armada tidak boleh basi di proses yang hidup lama
# ---------------------------------------------------------------------------


def test_potret_armada_dibuang_saat_data_bertambah(batch):
    """predict.py meng-cache potret armada per proses dan tidak memeriksa
    ulang batas data. Di server yang hidup berhari-hari itu membuat 3 fitur
    kondisi armada beku sementara 18 fitur lain ikut segar."""
    data_state.reset()
    data_state.current_data_end()
    failure_model._fleet_snapshot(batch.data_end)
    assert failure_model._FLEET is not None

    # Paksa pemeriksaan berikutnya melihat batas data yang berbeda.
    data_state._data_end = pd.Timestamp("2000-01-01")
    data_state._checked_at = 0.0
    generation_before = data_state.generation()
    data_state.current_data_end()

    assert failure_model._FLEET is None, "potret armada basi tidak dibuang"
    assert data_state.generation() > generation_before


def test_hasil_batch_ditandai_basi_saat_data_bertambah(batch):
    """Cache batch harus ikut kedaluwarsa karena DATA berubah, bukan hanya
    karena umurnya habis."""
    assert not batch.is_stale(batch.generation)
    assert batch.is_stale(batch.generation + 1)


def test_batas_data_tidak_ditanyakan_ulang_setiap_saat(count_connections):
    """Pemeriksaan kesegaran dipanggil di setiap request, jadi hasilnya harus
    ditahan - kalau tidak, ia sendiri yang jadi beban."""
    data_state.reset()
    data_state.current_data_end()
    first = count_connections["n"]
    for _ in range(5):
        data_state.current_data_end()
    assert count_connections["n"] == first


# ---------------------------------------------------------------------------
# Pembacaan yang terduplikasi dalam satu request
# ---------------------------------------------------------------------------


def test_assessment_tidak_membaca_hal_yang_sama_berulang(
    count_connections, scorable_item
):
    """predict() dan predict_scrap() membaca batas data, siklus, dan event yang
    sama persis. Tanpa penyatuan, satu assessment membuka 9 koneksi."""
    data_state.reset()
    count_connections["n"] = 0
    prediction_service.get_part_assessment(scorable_item, include_explanation=True)
    assert count_connections["n"] <= 4, (
        f"{count_connections['n']} koneksi untuk satu assessment - "
        "pembacaan berulang tidak tersatukan"
    )


def test_cache_hanya_hidup_di_dalam_scope(scorable_item):
    """Di luar request scope, data_reader harus berperilaku seperti semula -
    supaya train.py dan predict.py dari terminal tidak terpengaruh."""
    query_cache.install()
    assert query_cache.reads_in_scope() == 0
    with query_cache.request_scope():
        data_reader.get_events(scorable_item)
        assert query_cache.reads_in_scope() == 1
    assert query_cache.reads_in_scope() == 0


def test_cache_tidak_bertahan_antar_request(scorable_item):
    with query_cache.request_scope():
        data_reader.get_events(scorable_item)
        inside = query_cache.reads_in_scope()
    with query_cache.request_scope():
        assert query_cache.reads_in_scope() == 0
    assert inside == 1


def test_hasil_dengan_dan_tanpa_cache_identik(scorable_item):
    """Penghematan pembacaan tidak boleh mengubah satu angka pun."""
    cached = prediction_service.get_part_assessment(scorable_item, include_explanation=False)

    # Jalankan lagi tanpa scope sama sekali, langsung lewat ML core.
    direct_failure = failure_model.predict(scorable_item)
    for key, value in direct_failure.items():
        assert cached["failure"][key] == value, key


def test_argumen_berbeda_tidak_saling_menimpa(scorable_item, batch):
    """Kunci cache harus memasukkan argumen - kalau tidak, get_events(item)
    bisa menjawab get_events() untuk seluruh armada."""
    with query_cache.request_scope():
        one = data_reader.get_events(scorable_item)
        again = data_reader.get_events(scorable_item)
        assert one is again
        assert set(one["item_identifier_clean"].unique()) == {scorable_item}


# ---------------------------------------------------------------------------
# Penjelasan faktor risiko
# ---------------------------------------------------------------------------


def test_penjelasan_dari_batch_sama_dengan_yang_dihitung_langsung(batch, scorable_item):
    """Halaman detail memakai fitur hasil batch kalau tersedia. Nilainya harus
    sama persis dengan yang dibangun untuk satu PART."""
    from api.services import explanation

    from_batch = prediction_service._feature_row(scorable_item)
    direct = prediction_service._active_snapshot(scorable_item).iloc[0]

    for column in explanation.SOURCE_COLUMNS:
        left, right = from_batch[column], direct[column]
        if isinstance(left, float) and pd.isna(left):
            assert pd.isna(right), column
        else:
            assert left == right, f"{column}: batch={left!r} langsung={right!r}"


def test_penjelasan_tidak_memicu_batch_saat_cache_kosong(scorable_item):
    """Menjelaskan satu PART tidak boleh memaksa seluruh armada diskor."""
    from api.services import batch_service

    saved = batch_service._CACHE
    batch_service._CACHE = None
    try:
        result = prediction_service.explain(scorable_item)
        assert result["factors"]
        assert batch_service._CACHE is None, "penjelasan satu PART memicu batch penuh"
    finally:
        batch_service._CACHE = saved
