"""Prediksi untuk SATU PART, siap dipakai lapisan HTTP.

Membungkus predict.py dan predict_scrap.py apa adanya - tidak ada fitur yang
dihitung ulang dan tidak ada ambang yang ditentukan di sini. Tugas modul ini:

- memanggil model yang benar dan mengembalikan hasilnya utuh;
- membedakan "PART tidak ada" dari "PART ada tapi tidak bisa diskor", karena
  ML core melempar satu jenis kesalahan yang sama untuk keduanya;
- menggabungkan hasil kerusakan + scrap + rekomendasi + faktor risiko menjadi
  satu jawaban;
- menyembunyikan detail database dari pemanggil;
- menjaga satu request hanya membaca database seperlunya (lihat query_cache).
"""

from __future__ import annotations

import pandas as pd
import psycopg

import config
import data_reader
import feature_builder
import predict as failure_model
import predict_scrap as scrap_model
from api import data_state, query_cache
from api.errors import DataSourceUnavailable, PartNotFound, PartNotScorable
from api.services import (
    batch_service,
    explanation,
    history_service,
    recommendation_service,
)


def _guard(call, *args, **kwargs):
    """Jalankan panggilan ke ML core, ubah kegagalan database jadi kesalahan
    yang aman dibagikan ke client (tanpa DSN, kredensial, atau SQL)."""
    try:
        return call(*args, **kwargs)
    except psycopg.Error as error:
        raise DataSourceUnavailable(
            f"Database tidak bisa dibaca ({type(error).__name__})."
        ) from error


def _exists(item_id: str) -> bool:
    """Apakah PART ini benar-benar ada di database.

    Hanya dipanggil di jalur kesalahan: ML core memakai satu jenis kesalahan
    (ItemNotScorable) untuk "tidak ada" maupun "ada tapi tidak bisa dinilai",
    dan status HTTP-nya harus berbeda.
    """
    return not _guard(data_reader.get_events, item_id).empty


def _translate(item_id: str, error: Exception) -> Exception:
    if not _exists(item_id):
        return PartNotFound(item_id)

    # ML core memakai kalimat "tidak ditemukan di database" untuk dua hal:
    # PART yang benar-benar tidak ada, dan PART yang ada tetapi tidak punya
    # siklus pemasangan sebagai PART (mis. tercatat sebagai unit/mesin, atau
    # pemasangannya tidak pernah terekam). Di titik ini kita SUDAH membuktikan
    # PART-nya ada, jadi kalimat itu pasti keliru dan diganti - alasan lain
    # dari ML core, seperti "sedang tidak terpasang", tetap dipakai apa adanya.
    reason = str(error)
    if "tidak ditemukan" in reason:
        reason = (
            f"PART '{item_id}' ada di catatan, tetapi tidak punya siklus "
            "pemasangan sebagai PART yang bisa dinilai model."
        )
    return PartNotScorable(item_id, reason)


def predict_failure(item_id: str) -> dict:
    """Risiko kerusakan 30/60/90/120 hari untuk satu PART."""
    with query_cache.request_scope():
        data_state.current_data_end()
        try:
            return _guard(failure_model.predict, item_id)
        except failure_model.ItemNotScorable as error:
            raise _translate(item_id, error) from error


def predict_scrap(item_id: str) -> dict:
    """Kalau PART ini rusak, seberapa besar kemungkinan tidak bisa diperbaiki.

    Angkanya BERSYARAT terhadap kerusakan - bukan peluang PART ini rusak.
    """
    with query_cache.request_scope():
        data_state.current_data_end()
        try:
            return _guard(scrap_model.predict_scrap, item_id)
        except scrap_model.ItemNotScorable as error:
            raise _translate(item_id, error) from error


def get_part_assessment(item_id: str, include_explanation: bool = True) -> dict:
    """Gabungan kedua model + rekomendasi tindakan untuk satu PART.

    Seluruhnya dikerjakan dalam satu request scope: kedua model membaca batas
    waktu data, siklus, dan event yang sama persis, jadi pembacaannya cukup
    sekali walau dipakai dua kali.

    Risiko kerusakan WAJIB ada: kalau PART tidak bisa diskor model kerusakan,
    tidak ada penilaian yang bisa diberikan. Risiko scrap boleh kosong - PART
    yang riwayatnya belum cukup tetap dapat penilaian, hanya tanpa sumbu scrap
    (rekomendasinya menyesuaikan, bukan menebak).
    """
    with query_cache.request_scope():
        data_state.current_data_end()
        failure = predict_failure(item_id)

        try:
            scrap = predict_scrap(item_id)
        except PartNotScorable:
            scrap = None

        scrap_level = scrap["scrap_risk_level"] if scrap else None
        horizon = config.TARGET_HORIZON_DAYS
        assessment = {
            "item_id": failure["item_id"],
            "status": "SCORED",
            "as_of": failure["as_of"],
            "failure": failure,
            "scrap": scrap,
            # Rumus sama dengan predict_scrap.predict_death_risk(): peluang PART
            # benar-benar MATI = peluang rusak x peluang tidak bisa diperbaiki.
            # Dihitung di sini supaya kedua model tidak perlu dijalankan dua kali.
            f"death_probability_{horizon}d": (
                round(failure[f"failure_probability_{horizon}d"] * scrap["scrap_probability"], 5)
                if scrap
                else None
            ),
            "recommendation": recommendation_service.recommend(
                failure["risk_level"], scrap_level
            ),
            "replacement_candidate": recommendation_service.is_replacement_candidate(
                failure["risk_level"], scrap_level
            ),
            "model_version": {
                "failure": failure["model_version"],
                "scrap": scrap["model_version"] if scrap else None,
            },
        }

        if include_explanation:
            assessment["explanation"] = explain(item_id)
        return assessment


def explain(item_id: str) -> dict:
    """Faktor risiko satu PART, dari nilai fitur yang benar-benar dipakai model."""
    _, _, metadata = failure_model._load_model()
    row = _feature_row(item_id)
    factors = explanation.risk_factors(row)
    notes = [explanation.FAILURE_HISTORY_NOTE]
    if any(factor["code"].endswith("CORRECTIVE_MAINTENANCE") or
           factor["code"] == "CORRECTIVE_HISTORY" for factor in factors):
        notes.append(explanation.CORRECTIVE_NOTE)
    return {
        "disclaimer": explanation.DISCLAIMER,
        "factors": factors,
        "notes": notes,
        "caveats": explanation.caveats(row, metadata["part_model_support"]),
    }


def _feature_row(item_id: str) -> pd.Series:
    """Nilai fitur mentah satu PART.

    Diambil dari hasil batch scoring kalau ada: fitur untuk SELURUH PART aktif
    sudah dihitung di sana, dan hasil batch dijamin dihitung pada batas waktu
    data yang berlaku sekarang (lihat data_state), jadi memakainya tidak
    membuat halaman detail menampilkan angka yang lebih tua daripada
    probabilitasnya.

    Kalau batch belum pernah dijalankan, snapshot-nya dibangun untuk satu PART
    saja - jauh lebih murah daripada memaksa seluruh armada diskor hanya untuk
    menjelaskan satu PART.
    """
    cached = batch_service.cached_scores()
    if cached is not None and not cached.is_stale(data_state.generation()):
        # Normalisasi yang sama dengan yang dipakai data_reader saat
        # mencocokkan identifier, supaya "part-a" tetap ketemu.
        for key in (item_id, data_reader._normalize(item_id)):
            if key in cached.snapshot.index:
                return _single_row(cached.snapshot.loc[key])

    return _active_snapshot(item_id).iloc[0]


def _single_row(selection: pd.Series | pd.DataFrame) -> pd.Series:
    """Satu PART aktif seharusnya punya satu baris; kalau ternyata tidak,
    ambil yang pertama daripada meledak di tengah penjelasan."""
    return selection.iloc[0] if isinstance(selection, pd.DataFrame) else selection


def item_history(item_id: str) -> dict:
    """Tanggal kerusakan dan lokasi yang pernah tercatat untuk satu PART.

    Baris event apa adanya - bukan dihitung ulang - jadi jumlahnya selalu
    cocok dengan faktor risiko yang ditampilkan di /assessment (keduanya
    memakai definisi is_failure_onset yang sama dari data_reader.py).
    """
    with query_cache.request_scope():
        data_state.current_data_end()
        events = _guard(data_reader.get_events, item_id)
        if events.empty:
            raise PartNotFound(item_id)
        return {
            "item_id": item_id,
            "failures": history_service.failure_history(events),
            "locations": history_service.location_history(events),
        }


def _active_snapshot(item_id: str) -> pd.DataFrame:
    """Baris fitur mentah PART yang sedang terpasang.

    Urutan panggilannya sama persis seperti predict() - fungsi feature_builder
    yang sama, tanpa perhitungan tandingan. Diperlukan karena predict() hanya
    mengembalikan probabilitas, sementara halaman detail perlu angka mentah di
    baliknya.
    """
    with query_cache.request_scope():
        data_end = _guard(data_reader.get_dataset_max_event_on)
        cycles = _guard(data_reader.get_cycles, item_id, data_end)
        if cycles.empty:
            raise _translate(item_id, LookupError(f"PART '{item_id}' tidak ditemukan."))

        snapshot = feature_builder.current_observations(cycles)
        if snapshot.empty:
            raise PartNotScorable(
                item_id,
                f"PART '{item_id}' sedang tidak terpasang, jadi tidak ada fitur "
                "kondisi terkini yang bisa dijelaskan.",
            )
        snapshot = feature_builder.attach_history(
            snapshot, _guard(data_reader.get_events, item_id)
        )
        return feature_builder.attach_fleet_snapshot(
            snapshot, failure_model._fleet_snapshot(data_end)
        )
