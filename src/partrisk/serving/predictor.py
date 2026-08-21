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

from partrisk import config
from partrisk.data import reader as data_reader
from partrisk.features import failure as feature_builder
from partrisk.predict import failure as failure_model
from partrisk.predict import risk as death_risk
from partrisk.predict import scrap as scrap_model
from partrisk.predict import survival as predict_survival
from partrisk.serving import batch_predictor, data_state, explanation, history, query_cache, recommendation
from partrisk.serving.errors import DataSourceUnavailable, PartNotFound, PartNotScorable


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


def _survival_advisory_fields(item_id: str) -> dict:
    """Field ADVISORY (median_days_to_failure, days_until_survival_90pct,
    survival_curve, ...) dari model survival event-based - mode aditif,
    TIDAK PERNAH menggagalkan assessment utama (CatBoost tetap sumber
    failure_probability_*/risk_level). Kegagalan APA PUN di sini (model
    belum dilatih, PART tidak scorable model survival meski scorable
    CatBoost, dll) menghasilkan field kosong dengan alasan, bukan exception
    yang menjalar ke get_part_assessment()."""
    try:
        result = predict_survival.predict(item_id)
    except predict_survival.ItemNotScorable as error:
        return {
            "median_days_to_failure": None,
            "median_days_to_failure_basis": f"model survival: {error}",
            "days_until_survival_90pct": None,
            "survival_curve": None,
            "curve_step_days": None,
            "curve_horizon_days": None,
            "curve_is_calibrated": False,
        }
    except (Exception, SystemExit) as error:  # noqa: BLE001 - lihat docstring: advisory, tidak boleh menjalar
        return {
            "median_days_to_failure": None,
            "median_days_to_failure_basis": f"model survival tidak tersedia ({error})",
            "days_until_survival_90pct": None,
            "survival_curve": None,
            "curve_step_days": None,
            "curve_horizon_days": None,
            "curve_is_calibrated": False,
        }
    curve = result["estimated_survival_curve_from_now"]
    return {
        "median_days_to_failure": result["median_days_remaining_from_now"],
        "median_days_to_failure_basis": (
            None if result["median_days_remaining_from_now"] is not None
            else "S(t) belum turun sampai separuh dalam rentang follow-up training - tidak diekstrapolasi"
        ),
        "days_until_survival_90pct": result["days_until_survival_90pct_from_now"],
        "survival_curve": curve,
        "curve_step_days": predict_survival.CURVE_STEP_DAYS,
        "curve_horizon_days": curve[-1]["days_from_now"] if curve else None,
        "curve_is_calibrated": False,
    }


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
        failure.update(_survival_advisory_fields(item_id))

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
            # Dihitung di sini (bukan panggil predict_scrap.predict_death_risk())
            # supaya kedua model tidak perlu dijalankan dua kali.
            f"death_probability_{horizon}d": (
                death_risk.death_probability(
                    failure[f"failure_probability_{horizon}d"], scrap["scrap_probability"]
                )
                if scrap
                else None
            ),
            "recommendation": recommendation.recommend(
                failure["risk_level"], scrap_level
            ),
            "replacement_candidate": recommendation.is_replacement_candidate(
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
    _, _, metadata = failure_model.load_model()
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
    cached = batch_predictor.cached_scores()
    if cached is not None and not cached.is_stale(data_state.generation()):
        # Normalisasi yang sama dengan yang dipakai data_reader saat
        # mencocokkan identifier, supaya "part-a" tetap ketemu.
        for key in (item_id, data_reader.normalize(item_id)):
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
            "failures": history.failure_history(events),
            "locations": history.location_history(events),
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
            snapshot, failure_model.fleet_snapshot(data_end)
        )
